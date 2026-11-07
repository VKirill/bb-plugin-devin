import {
  createBridgeIo,
  experimental_defineProviderBridge,
  hostDaemonAcpLaunchSpecSchema,
  type ProviderBridgeEntry,
  type ReasoningLevel,
  type ServiceTier,
} from "@get-bb/plugin-sdk/provider-bridge";
import { experimental_acpProviderBridge } from "@get-bb/plugin-sdk/provider-bridge/acp";
import {
  discoverDevinModelOption,
  type DevinAcpLaunchSpec,
  type DevinModelDiscovery,
} from "./lib/devin-acp-discovery";
import {
  buildDevinModelCatalog,
  resolveDevinVariantUid,
  splitDevinPrimaryModels,
  type DevinModelCatalog,
  freeDevinModels,
} from "./lib/devin-model-catalog";
import {
  hostLaunchSpecDeps,
  localizeDevinLaunchSpec,
} from "./lib/host-launch-spec";

type BridgeIo = ReturnType<typeof createBridgeIo>;

interface DevinBridgeDeps {
  inner: ProviderBridgeEntry;
  io: BridgeIo;
  discover: (
    spec: DevinAcpLaunchSpec,
    cwd: string,
  ) => Promise<DevinModelDiscovery | null>;
  localizeLaunchSpec?: <T extends DevinAcpLaunchSpec>(spec: T) => T;
  /** Runs `run` with the inner bridge's stdout lines passed through `rewrite` (null drops a line). */
  captureOutput?: (rewrite: (line: string) => string | null, run: () => void) => void;
}

/**
 * VK core: a thread with a required session policy starts only on a bridge whose initialize declares
 * `experimental_vkRequiredSessionPolicy: 1` (else `vk_required_session_policy_bridge_unsupported`, every Lane Pilot
 * writer on Devin since 2026-11-05). The SDK's ACP bridge this plugin wraps predates it. What the core's own ACP bridge
 * enforces for a generic ACP agent is one thing — the BB skill list without `bbSkillsDenied` — so this wrapper does that
 * and declares the capability; the core does the rest (plugins, instructions, tools).
 */
const VK_REQUIRED_SESSION_POLICY_CAPABILITY = "experimental_vkRequiredSessionPolicy";

interface SkillRootWire {
  id: string;
  path: string;
  skills: Array<{ name: string; description?: string }>;
}

function deniedBbSkills(providerOptions: Record<string, unknown> | undefined): Set<string> {
  const policy = providerOptions?.["vkSessionPolicy"];
  const denied = policy && typeof policy === "object" ? (policy as { bbSkillsDenied?: unknown }).bbSkillsDenied : undefined;
  return new Set(Array.isArray(denied) ? denied.filter((name): name is string => typeof name === "string") : []);
}

function captureStdout(rewrite: (line: string) => string | null, run: () => void): void {
  const write = process.stdout.write;
  process.stdout.write = ((chunk: string | Uint8Array, ...rest: unknown[]) => {
    const out = typeof chunk === "string" ? rewrite(chunk) : chunk;
    if (out === null) return true;
    return (write as (...args: unknown[]) => boolean).call(process.stdout, out, ...rest);
  }) as typeof process.stdout.write;
  try {
    run();
  } finally {
    process.stdout.write = write;
  }
}

function lineId(line: string): unknown {
  try {
    return (JSON.parse(line) as { id?: unknown }).id;
  } catch {
    return undefined;
  }
}

const SESSION_OPTIONS_METHODS = new Set([
  "thread/start",
  "thread/resume",
  "thread/fork",
]);

const ACP_DEFAULT_MODEL_ID = "acp-default";

interface JsonRpcRequestMessage {
  jsonrpc?: string;
  id?: number | string;
  method?: string;
  params?: {
    cwd?: string;
    providerOptions?: Record<string, unknown>;
    options?: {
      model?: string;
      serviceTier?: ServiceTier;
      reasoningLevel?: ReasoningLevel;
      providerOptions?: Record<string, unknown>;
      [key: string]: unknown;
    };
    [key: string]: unknown;
  };
}

function decodeLaunchSpec(
  providerOptions: Record<string, unknown> | undefined,
): DevinAcpLaunchSpec | null {
  const parsed = hostDaemonAcpLaunchSpecSchema.safeParse(
    providerOptions?.["acpLaunchSpec"],
  );
  return parsed.success ? parsed.data : null;
}

export function createDevinProviderBridge(deps: DevinBridgeDeps) {
  const { inner, io, discover } = deps;
  const localize = deps.localizeLaunchSpec;
  const capture = deps.captureOutput ?? captureStdout;
  let skillRoots: SkillRootWire[] | null = null;
  let synthetic = 0;

  function forwardInitialize(request: JsonRpcRequestMessage, line: string): void {
    capture((out) => {
      if (lineId(out) !== request.id) return out;
      const message = JSON.parse(out) as { result?: { capabilities?: Record<string, unknown> } };
      if (!message.result?.capabilities) return out;
      message.result.capabilities[VK_REQUIRED_SESSION_POLICY_CAPABILITY] = 1;
      return `${JSON.stringify(message)}\n`;
    }, () => inner.handleLine(line));
  }

  // The inner bridge keeps one skill list for every session and copies it into a session while it starts, before its
  // first await (SDK 0.4.84 provider-bridge-acp, thread/start). So a session with denied skills starts between a
  // filtered skills/configure and the full one again, all in one synchronous run; their two answers are dropped.
  // ponytail: tied to that synchronous copy; move to the core's ACP bridge when the SDK ships it.
  function forwardSession(line: string, providerOptions: Record<string, unknown> | undefined): void {
    const denied = deniedBbSkills(providerOptions);
    if (denied.size === 0 || !skillRoots) {
      inner.handleLine(line);
      return;
    }
    const filtered = skillRoots.map((root) => ({ ...root, skills: root.skills.filter((skill) => !denied.has(skill.name)) }));
    const ids = new Set([`devin-vk-skills-${++synthetic}`, `devin-vk-skills-${++synthetic}`]);
    const [narrowId, restoreId] = [...ids];
    const configure = (id: string, roots: SkillRootWire[]) =>
      JSON.stringify({ jsonrpc: "2.0", id, method: "skills/configure", params: { roots } });
    capture((out) => (ids.has(lineId(out) as string) ? null : out), () => {
      inner.handleLine(configure(narrowId!, filtered));
      inner.handleLine(line);
      inner.handleLine(configure(restoreId!, skillRoots!));
    });
  }

  // Rewrite server-built launch specs to this host's paths before anything
  // (discovery, the inner ACP bridge, health checks) spawns them.
  function localizeRequest(
    request: JsonRpcRequestMessage,
    line: string,
  ): { request: JsonRpcRequestMessage; line: string } {
    if (!localize || !request.params) {
      return { request, line };
    }
    const params = request.params;
    let changed = false;
    const rewrite = (
      providerOptions: Record<string, unknown> | undefined,
    ): Record<string, unknown> | undefined => {
      const spec = decodeLaunchSpec(providerOptions);
      if (!providerOptions || !spec) {
        return providerOptions;
      }
      const localized = localize(spec);
      if (localized === spec) {
        return providerOptions;
      }
      changed = true;
      return { ...providerOptions, acpLaunchSpec: localized };
    };
    const providerOptions = rewrite(params.providerOptions);
    const optionsProviderOptions = rewrite(params.options?.providerOptions);
    if (!changed) {
      return { request, line };
    }
    const next: JsonRpcRequestMessage = {
      ...request,
      params: {
        ...params,
        ...(params.providerOptions ? { providerOptions } : {}),
        ...(params.options
          ? {
              options: {
                ...params.options,
                ...(params.options.providerOptions
                  ? { providerOptions: optionsProviderOptions }
                  : {}),
              },
            }
          : {}),
      },
    };
    return { request: next, line: JSON.stringify(next) };
  }
  let catalog: DevinModelCatalog | null = null;
  let catalogKey: string | undefined;

  async function ensureCatalog(
    spec: DevinAcpLaunchSpec,
    cwd: string,
  ): Promise<DevinModelCatalog | null> {
    const key = JSON.stringify({ command: spec.command, args: spec.args, cwd });
    if (catalog && catalogKey === key) {
      return catalog;
    }
    const discovery = await discover(spec, cwd);
    if (!discovery) {
      return null;
    }
    catalog = buildDevinModelCatalog(
      freeDevinModels(discovery.options),
      discovery.currentValue,
      discovery.thoughtLevelsByModel,
    );
    catalogKey = key;
    return catalog;
  }

  async function handleModelList(
    request: JsonRpcRequestMessage,
    line: string,
  ): Promise<void> {
    const spec = decodeLaunchSpec(request.params?.providerOptions);
    const cwd = request.params?.cwd ?? process.cwd();
    const discovery = spec ? await discover(spec, cwd) : null;
    if (!spec || !discovery) {
      inner.handleLine(line);
      return;
    }
    const key = JSON.stringify({ command: spec.command, args: spec.args, cwd });
    let resolvedCatalog = catalog;
    if (catalogKey !== key || !resolvedCatalog) {
      resolvedCatalog = buildDevinModelCatalog(
        freeDevinModels(discovery.options),
        discovery.currentValue,
        discovery.thoughtLevelsByModel,
      );
      catalog = resolvedCatalog;
      catalogKey = key;
    }
    const primary = request.params?.providerOptions?.["primaryModels"];
    const split = splitDevinPrimaryModels(
      resolvedCatalog.models,
      Array.isArray(primary)
        ? primary.filter((entry): entry is string => typeof entry === "string")
        : undefined,
    );
    if (request.id === undefined) {
      return;
    }
    io.sendResult(request.id, split);
  }

  async function rewriteAndForward(
    request: JsonRpcRequestMessage,
    line: string,
  ): Promise<void> {
    const sessionOptions = request.params?.options?.providerOptions;
    try {
      const options = request.params?.options;
      const spec = decodeLaunchSpec(
        options?.providerOptions ?? request.params?.providerOptions,
      );
      if (
        !options ||
        !spec ||
        options.model === undefined ||
        options.model === ACP_DEFAULT_MODEL_ID
      ) {
        forwardSession(line, sessionOptions);
        return;
      }
      const resolvedCatalog = await ensureCatalog(
        spec,
        request.params?.cwd ?? process.cwd(),
      );
      const resolved = resolvedCatalog
        ? resolveDevinVariantUid(resolvedCatalog, {
            model: options.model,
            reasoningLevel: options.reasoningLevel,
            serviceTier: options.serviceTier,
          })
        : undefined;
      if (resolved === undefined || resolved === options.model) {
        forwardSession(line, sessionOptions);
        return;
      }
      forwardSession(
        JSON.stringify({
          ...request,
          params: {
            ...request.params,
            options: { ...options, model: resolved },
          },
        }),
        sessionOptions,
      );
    } catch {
      forwardSession(line, sessionOptions);
    }
  }

  return experimental_defineProviderBridge({
    handleLine(line) {
      const trimmed = line.trim();
      if (!trimmed) {
        return;
      }
      let parsed: JsonRpcRequestMessage;
      try {
        parsed = JSON.parse(trimmed) as JsonRpcRequestMessage;
      } catch {
        inner.handleLine(line);
        return;
      }
      const { request, line: localLine } = localizeRequest(parsed, trimmed);
      if (request.method === "initialize" && request.id !== undefined) {
        forwardInitialize(request, localLine);
        return;
      }
      if (request.method === "skills/configure") {
        const roots = request.params?.["roots"];
        skillRoots = Array.isArray(roots) ? (roots as SkillRootWire[]) : null;
      }
      if (request.method === "model/list" && request.id !== undefined) {
        void handleModelList(request, localLine);
        return;
      }
      if (
        request.method !== undefined &&
        SESSION_OPTIONS_METHODS.has(request.method) &&
        request.id !== undefined
      ) {
        void rewriteAndForward(request, localLine);
        return;
      }
      inner.handleLine(localLine === trimmed ? line : localLine);
    },
    ...(inner.start ? { start: (ctx) => inner.start?.(ctx) } : {}),
    ...(inner.onClose ? { onClose: () => inner.onClose?.() } : {}),
    ...(inner.onSigterm ? { onSigterm: () => inner.onSigterm?.() } : {}),
    ...(inner.onSigint ? { onSigint: () => inner.onSigint?.() } : {}),
  });
}

export const experimental_providerBridge = createDevinProviderBridge({
  inner: experimental_acpProviderBridge,
  io: createBridgeIo(),
  discover: discoverDevinModelOption,
  localizeLaunchSpec: (spec) => localizeDevinLaunchSpec(spec, hostLaunchSpecDeps),
});
