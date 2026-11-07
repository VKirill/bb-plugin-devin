import { describe, expect, it } from "vitest";
import {
  createBridgeIo,
  type ProviderBridgeEntry,
} from "@get-bb/plugin-sdk/provider-bridge";
import { experimental_acpProviderBridge } from "@get-bb/plugin-sdk/provider-bridge/acp";
import { createDevinProviderBridge } from "../host";

const LAUNCH_SPEC = { displayName: "Devin", command: "devin", args: ["acp"], env: {} };
const ROOTS = [
  { id: "bb", path: "/skills", skills: [{ name: "writer-practices" }, { name: "seo-agent" }, { name: "ru-text" }] },
];

/** A fake inner bridge that answers like the SDK one: synchronously, through the captured output. */
function harness() {
  const forwarded: Array<{ method?: string; params?: { roots?: typeof ROOTS } }> = [];
  const out: string[] = [];
  let rewrite: ((line: string) => string | null) | null = null;
  const emit = (message: unknown) => {
    const line = `${JSON.stringify(message)}\n`;
    const next = rewrite ? rewrite(line) : line;
    if (next !== null) out.push(next);
  };
  const inner: ProviderBridgeEntry = {
    experimental_apiVersion: 1,
    handleLine: (line) => {
      const message = JSON.parse(line) as { id?: unknown; method?: string; params?: { roots?: typeof ROOTS } };
      forwarded.push(message);
      if (message.method === "initialize") emit({ jsonrpc: "2.0", id: message.id, result: { ok: true, capabilities: { fork: "tip" } } });
      if (message.method === "skills/configure") emit({ jsonrpc: "2.0", id: message.id, result: { ok: true } });
    },
  };
  const bridge = createDevinProviderBridge({
    inner,
    io: createBridgeIo({ write: () => {} }),
    discover: async () => null,
    captureOutput: (fn, run) => {
      rewrite = fn;
      try { run(); } finally { rewrite = null; }
    },
  });
  return { bridge, forwarded, out: () => out.map((line) => JSON.parse(line) as Record<string, any>) };
}

const flush = async () => { for (let i = 0; i < 20; i += 1) await new Promise((r) => setImmediate(r)); };

function threadStart(policy?: unknown) {
  return JSON.stringify({
    jsonrpc: "2.0", id: 9, method: "thread/start",
    params: { cwd: "/repo", options: { model: "acp-default", providerOptions: { acpLaunchSpec: LAUNCH_SPEC, ...(policy ? { vkRequiredSessionPolicy: 1, vkSessionPolicy: policy } : {}) } } },
  });
}

describe("required session policy on the Devin bridge", () => {
  it("declares the capability in its initialize answer", () => {
    const { bridge, out } = harness();
    bridge.handleLine(JSON.stringify({ jsonrpc: "2.0", id: 1, method: "initialize", params: {} }));
    expect(out()[0]?.result.capabilities).toEqual({ fork: "tip", experimental_vkRequiredSessionPolicy: 1 });
  });

  it("starts a session with denied BB skills on the narrowed list, then restores the full one, and drops both answers", async () => {
    const { bridge, forwarded, out } = harness();
    bridge.handleLine(JSON.stringify({ jsonrpc: "2.0", id: 2, method: "skills/configure", params: { roots: ROOTS } }));
    bridge.handleLine(threadStart({ required: true, bbSkillsDenied: ["seo-agent"] }));
    await flush();
    expect(forwarded.map((m) => m.method)).toEqual(["skills/configure", "skills/configure", "thread/start", "skills/configure"]);
    expect(forwarded[1]!.params!.roots![0]!.skills.map((s) => s.name)).toEqual(["writer-practices", "ru-text"]);
    expect(forwarded[3]!.params!.roots).toEqual(ROOTS);
    expect(out().map((m) => m.id)).toEqual([2]);
  });

  it("forwards a session without a policy untouched", async () => {
    const { bridge, forwarded } = harness();
    bridge.handleLine(JSON.stringify({ jsonrpc: "2.0", id: 2, method: "skills/configure", params: { roots: ROOTS } }));
    bridge.handleLine(threadStart());
    await flush();
    expect(forwarded.map((m) => m.method)).toEqual(["skills/configure", "thread/start"]);
  });

  it("the SDK's real ACP bridge answers initialize synchronously, so the capability reaches stdout", () => {
    const bridge = createDevinProviderBridge({ inner: experimental_acpProviderBridge, io: createBridgeIo({ write: () => {} }), discover: async () => null });
    const lines: string[] = [];
    const write = process.stdout.write;
    process.stdout.write = ((chunk: string) => { lines.push(chunk); return true; }) as typeof process.stdout.write;
    try {
      bridge.handleLine(JSON.stringify({ jsonrpc: "2.0", id: 1, method: "initialize", params: { protocolVersion: 1, client: { name: "bb", version: "1" } } }));
    } finally {
      process.stdout.write = write;
    }
    const answer = lines.map((l) => JSON.parse(l)).find((m) => m.id === 1);
    expect(answer?.result?.capabilities?.experimental_vkRequiredSessionPolicy).toBe(1);
  });
});
