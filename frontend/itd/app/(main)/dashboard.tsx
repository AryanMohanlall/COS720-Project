const stats = [
  { label: "Open incidents", value: "12", tone: "text-rose-600" },
  { label: "Pending approvals", value: "5", tone: "text-amber-600" },
  { label: "Resolved today", value: "28", tone: "text-emerald-600" },
];

const tasks = [
  "Review overnight alerts",
  "Approve access requests",
  "Validate backup completion",
  "Send daily operations summary",
];

export default function Dashboard() {
  return (
    <main className="min-h-screen bg-slate-100 px-6 py-8 text-slate-950">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8">
        <header className="flex flex-col gap-4 rounded-[2rem] bg-slate-950 px-8 py-8 text-white shadow-[0_24px_80px_-32px_rgba(15,23,42,0.6)] sm:flex-row sm:items-end sm:justify-between">
          <div className="space-y-2">
            <p className="text-sm uppercase tracking-[0.24em] text-sky-200">
              Dashboard
            </p>
            <h1 className="text-3xl font-semibold tracking-tight">
              ITD operations overview
            </h1>
            <p className="max-w-2xl text-sm leading-6 text-slate-300">
              Track current workload, monitor key activity, and keep the team
              aligned on today&apos;s priorities.
            </p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-200">
            Last sync: 08:45 SAST
          </div>
        </header>

        <section className="grid gap-4 md:grid-cols-3">
          {stats.map((stat) => (
            <article
              key={stat.label}
              className="rounded-[1.5rem] bg-white p-6 shadow-sm ring-1 ring-slate-200"
            >
              <p className="text-sm text-slate-500">{stat.label}</p>
              <p className={`mt-3 text-4xl font-semibold ${stat.tone}`}>
                {stat.value}
              </p>
            </article>
          ))}
        </section>

        <section className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
          <article className="rounded-[1.5rem] bg-white p-6 shadow-sm ring-1 ring-slate-200">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Today&apos;s priorities</h2>
              <span className="rounded-full bg-sky-100 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-sky-700">
                Focus
              </span>
            </div>
            <ul className="mt-6 space-y-3">
              {tasks.map((task, index) => (
                <li
                  key={task}
                  className="flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-3 text-sm text-slate-700"
                >
                  <span>{task}</span>
                  <span className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                    P{index + 1}
                  </span>
                </li>
              ))}
            </ul>
          </article>

          <article className="rounded-[1.5rem] bg-white p-6 shadow-sm ring-1 ring-slate-200">
            <h2 className="text-xl font-semibold">System health</h2>
            <div className="mt-6 space-y-4">
              <div className="rounded-2xl bg-emerald-50 px-4 py-4">
                <p className="text-sm text-emerald-800">Infrastructure</p>
                <p className="mt-1 text-2xl font-semibold text-emerald-950">
                  Stable
                </p>
              </div>
              <div className="rounded-2xl bg-amber-50 px-4 py-4">
                <p className="text-sm text-amber-800">Scheduled maintenance</p>
                <p className="mt-1 text-lg font-semibold text-amber-950">
                  Database patching at 18:00
                </p>
              </div>
            </div>
          </article>
        </section>
      </div>
    </main>
  );
}
