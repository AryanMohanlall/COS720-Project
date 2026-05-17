import Link from "next/link";

export default function Home() {
  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden bg-[#020b14] cyber-grid px-6 py-12">
      {/* Scan line */}
      <div className="scan pointer-events-none fixed inset-x-0 h-px bg-gradient-to-r from-transparent via-cyan-400/40 to-transparent" />

      {/* Corner brackets */}
      <div className="pointer-events-none absolute left-6 top-6 h-16 w-16 border-l-2 border-t-2 border-cyan-500/30" />
      <div className="pointer-events-none absolute right-6 top-6 h-16 w-16 border-r-2 border-t-2 border-cyan-500/30" />
      <div className="pointer-events-none absolute bottom-6 left-6 h-16 w-16 border-b-2 border-l-2 border-cyan-500/30" />
      <div className="pointer-events-none absolute bottom-6 right-6 h-16 w-16 border-b-2 border-r-2 border-cyan-500/30" />

      <section className="relative w-full max-w-4xl overflow-hidden rounded-sm border border-cyan-500/20 bg-slate-900/95 shadow-[0_0_80px_rgba(6,182,212,0.08)] backdrop-blur glow-cyan">
        {/* Top accent line */}
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-500 to-transparent" />
        {/* Bottom accent line */}
        <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-cyan-500/40 to-transparent" />

        <div className="grid sm:grid-cols-[1.15fr_0.85fr]">

          {/* ── Left: System Info ───────────────────────────── */}
          <div className="flex flex-col justify-between gap-8 border-r border-slate-700/40 bg-[#030e1a] px-8 py-10 sm:px-10 sm:py-12">
            <div className="space-y-6">
              {/* Classification badge */}
              <div className="inline-flex items-center gap-2.5">
                <span className="blink h-2 w-2 rounded-full bg-red-500" />
                <span className="font-mono text-[10px] font-bold uppercase tracking-[0.32em] text-red-400">
                  CLASSIFIED // RESTRICTED ACCESS
                </span>
              </div>

              <div className="space-y-2">
                <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-cyan-500/70">
                  // SYSTEM DESIGNATION
                </p>
                <h1 className="font-mono text-4xl font-black tracking-tight text-slate-100 flicker sm:text-5xl">
                  ITD PORTAL
                </h1>
                <p className="font-mono text-xs text-slate-500 uppercase tracking-widest">
                  INSIDER THREAT DETECTION SYSTEM
                </p>
              </div>

              <p className="font-mono text-sm leading-7 text-slate-400">
                Centralize incident monitoring, workflow approvals, and
                behavioral analysis across all organizational units.
              </p>
            </div>

            {/* Stats */}
            <div className="grid gap-3 text-sm sm:grid-cols-3">
              {(
                [
                  { value: "24/7", label: "MONITORING" },
                  { value: "1 HUB", label: "OPERATIONS" },
                  { value: "SECURE", label: "ACCESS" },
                ] as const
              ).map(({ value, label }) => (
                <div
                  key={label}
                  className="rounded-sm border border-cyan-500/15 bg-cyan-950/20 p-4"
                >
                  <p className="font-mono text-xl font-bold text-cyan-400">
                    {value}
                  </p>
                  <p className="mt-1 font-mono text-[10px] uppercase tracking-widest text-slate-500">
                    {label}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* ── Right: Auth Panel ───────────────────────────── */}
          <div className="flex items-center bg-slate-900 px-8 py-10 sm:px-10 sm:py-12">
            <div className="w-full space-y-7">
              <div className="space-y-2">
                <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-cyan-400">
                  // AUTHENTICATION REQUIRED
                </p>
                <h2 className="font-mono text-2xl font-black tracking-tight text-slate-100">
                  SECURE SIGN-IN
                </h2>
                <p className="font-mono text-xs leading-6 text-slate-500">
                  Authenticate via your organization&apos;s Google account to
                  access classified dashboard systems.
                </p>
              </div>

              <button
                type="button"
                className="flex w-full items-center justify-center gap-3 rounded-sm border border-slate-600 bg-slate-800 px-5 py-4 font-mono text-sm font-bold text-slate-100 transition hover:border-cyan-500/50 hover:bg-slate-700/80 hover:shadow-[0_0_16px_rgba(6,182,212,0.15)]"
              >
                <span className="flex h-7 w-7 items-center justify-center rounded-full bg-[conic-gradient(from_180deg,_#4285f4_0deg_90deg,_#34a853_90deg_180deg,_#fbbc05_180deg_270deg,_#ea4335_270deg_360deg)] font-black text-xs text-white">
                  G
                </span>
                AUTHENTICATE WITH GOOGLE
              </button>

              {/* Dev notice */}
              <div className="rounded-sm border border-amber-500/25 bg-amber-950/20 px-4 py-3 font-mono text-xs text-amber-400">
                <span className="font-bold">[!]</span> Frontend placeholder — connect button to Google OAuth when auth is wired up.
              </div>

              <div className="flex items-center justify-between border-t border-slate-700/50 pt-4 font-mono text-xs text-slate-500">
                <span className="uppercase tracking-widest">DEV BYPASS</span>
                <Link
                  href="/dashboard"
                  className="font-bold text-cyan-400 transition hover:text-cyan-300"
                >
                  OPEN DASHBOARD →
                </Link>
              </div>
            </div>
          </div>

        </div>
      </section>
    </main>
  );
}
