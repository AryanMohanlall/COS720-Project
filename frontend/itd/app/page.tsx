import Link from "next/link";

export default function Home() {
  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden bg-[radial-gradient(circle_at_top,_#fef3c7_0%,_#fff7ed_30%,_#f8fafc_70%)] px-6 py-12 text-slate-950">
      <div className="absolute left-[-8rem] top-[-6rem] h-56 w-56 rounded-full bg-amber-300/30 blur-3xl" />
      <div className="absolute bottom-[-7rem] right-[-5rem] h-64 w-64 rounded-full bg-sky-300/30 blur-3xl" />

      <section className="relative grid w-full max-w-5xl overflow-hidden rounded-[2rem] border border-white/70 bg-white/85 shadow-[0_40px_120px_-40px_rgba(15,23,42,0.35)] backdrop-blur sm:grid-cols-[1.1fr_0.9fr]">
        <div className="flex flex-col justify-between gap-10 bg-slate-950 px-8 py-10 text-white sm:px-10 sm:py-12">
          <div className="space-y-5">
            <span className="inline-flex w-fit rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.28em] text-amber-200">
              ITD Portal
            </span>
            <div className="space-y-4">
              <h1 className="max-w-md text-4xl font-semibold tracking-tight sm:text-5xl">
                Sign in with Google to access the operations dashboard.
              </h1>
              <p className="max-w-lg text-sm leading-7 text-slate-300 sm:text-base">
                Centralize incident monitoring, workflow approvals, and daily
                status updates in one internal workspace.
              </p>
            </div>
          </div>

          <div className="grid gap-4 text-sm text-slate-300 sm:grid-cols-3">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="text-2xl font-semibold text-white">24/7</p>
              <p className="mt-1">System visibility across teams.</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="text-2xl font-semibold text-white">1 Hub</p>
              <p className="mt-1">Shared access for IT and operations.</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="text-2xl font-semibold text-white">Secure</p>
              <p className="mt-1">Google SSO for controlled sign-in.</p>
            </div>
          </div>
        </div>

        <div className="flex items-center bg-white px-8 py-10 sm:px-10 sm:py-12">
          <div className="w-full space-y-8">
            <div className="space-y-2">
              <p className="text-sm font-medium uppercase tracking-[0.24em] text-slate-500">
                Welcome back
              </p>
              <h2 className="text-3xl font-semibold tracking-tight text-slate-950">
                Continue with your work account
              </h2>
              <p className="text-sm leading-6 text-slate-600">
                Use your organization&apos;s Google account to authenticate and
                continue to the internal dashboard.
              </p>
            </div>

            <button
              type="button"
              className="flex w-full items-center justify-center gap-3 rounded-2xl border border-slate-200 bg-white px-5 py-4 text-sm font-semibold text-slate-900 shadow-sm transition hover:-translate-y-0.5 hover:border-slate-300 hover:shadow-md"
            >
              <span className="flex h-8 w-8 items-center justify-center rounded-full bg-[conic-gradient(from_180deg_at_50%_50%,_#4285f4_0deg,_#4285f4_90deg,_#34a853_90deg,_#34a853_180deg,_#fbbc05_180deg,_#fbbc05_270deg,_#ea4335_270deg,_#ea4335_360deg)] text-xs font-bold text-white">
                G
              </span>
              Sign in with Google
            </button>

            <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950">
              Frontend placeholder: connect this button to your Google OAuth
              flow when auth is wired up.
            </div>

            <div className="flex items-center justify-between border-t border-slate-200 pt-4 text-sm text-slate-500">
              <span>Need a preview first?</span>
              <Link
                href="/dashboard"
                className="font-semibold text-slate-900 transition hover:text-sky-700"
              >
                Open dashboard
              </Link>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
