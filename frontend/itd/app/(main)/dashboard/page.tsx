"use client";

import { useEffect, useMemo, useState } from "react";

import {
  API_BASE_URL,
  ApiError,
  MODEL_NAMES,
  type MetricsResponse,
  type ModelName,
  type ModelsStatusResponse,
  type PredictionFeatures,
  type PredictionResponse,
  api,
} from "@/services/api";

const MODEL_LABELS: Record<ModelName, string> = {
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  random_forest: "Random Forest",
};

const SAMPLE_FEATURES: PredictionFeatures = {
  employee_department: "IT",
  employee_campus: "Campus A",
  employee_position: "Analyst",
  employee_seniority_years: 3,
  is_contractor: 0,
  employee_classification: 2,
  has_foreign_citizenship: 0,
  has_criminal_record: 0,
  has_medical_history: 0,
  employee_origin_country: "South Africa",
  total_printed_pages: 24,
  num_printed_pages_off_hours: 0,
  total_files_burned: 0,
  burned_from_other: 0,
  is_abroad: 0,
  trip_day_number: 0,
  hostility_country_level: 1,
  num_entries: 4,
  num_unique_campus: 1,
  late_exit_flag: 0,
  entry_during_weekend: 0,
};

const FEATURE_FIELDS = [
  { name: "employee_department", label: "Department", type: "text" },
  { name: "employee_campus", label: "Campus", type: "text" },
  { name: "employee_position", label: "Position", type: "text" },
  { name: "employee_seniority_years", label: "Seniority years", type: "number" },
  { name: "is_contractor", label: "Is contractor", type: "number" },
  { name: "employee_classification", label: "Classification", type: "number" },
  { name: "has_foreign_citizenship", label: "Foreign citizenship", type: "number" },
  { name: "has_criminal_record", label: "Criminal record", type: "number" },
  { name: "has_medical_history", label: "Medical history", type: "number" },
  { name: "employee_origin_country", label: "Origin country", type: "text" },
  { name: "total_printed_pages", label: "Printed pages", type: "number" },
  { name: "num_printed_pages_off_hours", label: "Off-hours pages", type: "number" },
  { name: "total_files_burned", label: "Files burned", type: "number" },
  { name: "burned_from_other", label: "Burned from other", type: "number" },
  { name: "is_abroad", label: "Is abroad", type: "number" },
  { name: "trip_day_number", label: "Trip day number", type: "number" },
  { name: "hostility_country_level", label: "Hostility level", type: "number" },
  { name: "num_entries", label: "Entries", type: "number" },
  { name: "num_unique_campus", label: "Unique campuses", type: "number" },
  { name: "late_exit_flag", label: "Late exit", type: "number" },
  { name: "entry_during_weekend", label: "Weekend entry", type: "number" },
] as const;

const NUMERIC_FEATURES: ReadonlySet<string> = new Set(
  FEATURE_FIELDS.filter((field) => field.type === "number").map(
    (field) => field.name,
  ),
);

const FEATURE_LABELS = FEATURE_FIELDS.reduce<Record<string, string>>(
  (labels, field) => ({
    ...labels,
    [field.name]: field.label,
  }),
  {},
);

function formatPercent(value: number | undefined) {
  if (typeof value !== "number") return "n/a";
  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value: number | undefined) {
  if (typeof value !== "number") return "n/a";
  return value.toLocaleString();
}

function formatShapValue(value: number) {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(4)}`;
}

function formatFeatureValue(value: unknown) {
  if (value === null || value === undefined || value === "") return "blank";
  if (typeof value === "number") return value.toLocaleString();
  if (typeof value === "string" || typeof value === "boolean") return value.toString();
  return JSON.stringify(value);
}

function readableFeatureName(feature: string) {
  return FEATURE_LABELS[feature] ?? feature.replaceAll("_", " ");
}

function shapMagnitude(value: number) {
  const absoluteValue = Math.abs(value);
  if (absoluteValue >= 1) return "strong";
  if (absoluteValue >= 0.25) return "moderate";
  return "small";
}

function shapSentence(contribution: {
  feature: string;
  value: unknown;
  shap_value: number;
  direction: "increases_risk" | "decreases_risk";
}) {
  const feature = readableFeatureName(contribution.feature);
  const value = formatFeatureValue(contribution.value);
  const direction =
    contribution.direction === "increases_risk"
      ? "pushed the score toward a malicious prediction"
      : "pushed the score away from a malicious prediction";
  return `${feature} (${value}) had a ${shapMagnitude(contribution.shap_value)} effect and ${direction}.`;
}

function shapSummary(prediction: PredictionResponse) {
  const topContributions = prediction.explanation?.top_contributions ?? [];
  const riskDrivers = topContributions.filter((c) => c.direction === "increases_risk");
  const protectiveDrivers = topContributions.filter((c) => c.direction === "decreases_risk");
  const strongestRisk = riskDrivers[0];
  const strongestProtective = protectiveDrivers[0];
  const predictedLabel = prediction.prediction === 1 ? "malicious" : "not malicious";

  if (!strongestRisk && !strongestProtective)
    return `The model predicted ${predictedLabel}, but no SHAP drivers were returned.`;

  if (strongestRisk && strongestProtective)
    return `The model predicted ${predictedLabel}. The strongest risk driver was ${readableFeatureName(strongestRisk.feature)}, while ${readableFeatureName(strongestProtective.feature)} most reduced the risk score.`;

  const strongestDriver = strongestRisk ?? strongestProtective;
  return `The model predicted ${predictedLabel}. The strongest driver was ${readableFeatureName(strongestDriver.feature)}, which ${strongestDriver.direction === "increases_risk" ? "increased" : "reduced"} the risk score.`;
}

function getErrorMessage(error: unknown) {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return "Request failed";
}

function coerceFeatureValue(name: string, value: string) {
  const trimmed = value.trim();
  if (trimmed === "") return null;
  if (!NUMERIC_FEATURES.has(name)) return trimmed;
  const numberValue = Number(trimmed);
  if (Number.isNaN(numberValue)) throw new Error(`${name} must be numeric.`);
  return numberValue;
}

function parseCsvRows(csv: string) {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let inQuotes = false;

  for (let index = 0; index < csv.length; index += 1) {
    const character = csv[index];
    const nextCharacter = csv[index + 1];

    if (character === '"' && inQuotes && nextCharacter === '"') {
      cell += '"';
      index += 1;
      continue;
    }
    if (character === '"') { inQuotes = !inQuotes; continue; }
    if (character === "," && !inQuotes) { row.push(cell); cell = ""; continue; }
    if ((character === "\n" || character === "\r") && !inQuotes) {
      if (character === "\r" && nextCharacter === "\n") index += 1;
      row.push(cell);
      if (row.some((v) => v.trim() !== "")) rows.push(row);
      row = []; cell = "";
      continue;
    }
    cell += character;
  }

  row.push(cell);
  if (row.some((v) => v.trim() !== "")) rows.push(row);
  return rows;
}

function featuresFromCsv(csv: string, currentFeatures: PredictionFeatures) {
  const rows = parseCsvRows(csv);
  if (rows.length < 2) throw new Error("CSV must include a header row and at least one data row.");

  const headers = rows[0].map((h) => h.trim());
  const firstDataRow = rows[1];
  const nextFeatures = { ...currentFeatures };

  headers.forEach((header, index) => {
    if (!FEATURE_FIELDS.some((field) => field.name === header)) return;
    nextFeatures[header] = coerceFeatureValue(header, firstDataRow[index] ?? "");
  });

  return nextFeatures;
}

/* ─── Component ──────────────────────────────────────────────────── */

export default function Dashboard() {
  const [selectedModel, setSelectedModel] = useState<ModelName>("xgboost");
  const [status, setStatus] = useState<ModelsStatusResponse | null>(null);
  const [metrics, setMetrics] = useState<Partial<Record<ModelName, MetricsResponse>>>({});
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [formFeatures, setFormFeatures] = useState<PredictionFeatures>(SAMPLE_FEATURES);
  const [csvInput, setCsvInput] = useState("");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    async function loadDashboard() {
      setLoading(true);
      setLoadError(null);
      try {
        const [modelStatus, metricResponses] = await Promise.all([
          api.getModelsStatus(),
          Promise.all(MODEL_NAMES.map((n) => api.getModelMetrics(n))),
        ]);
        if (!isMounted) return;
        setStatus(modelStatus);
        setMetrics(
          metricResponses.reduce<Partial<Record<ModelName, MetricsResponse>>>(
            (acc, r) => ({ ...acc, [r.model_name]: r }),
            {},
          ),
        );
      } catch (error) {
        if (isMounted) setLoadError(getErrorMessage(error));
      } finally {
        if (isMounted) setLoading(false);
      }
    }

    loadDashboard();
    return () => { isMounted = false; };
  }, []);

  const selectedMetrics = metrics[selectedModel]?.metrics;
  const selectedStatus = status?.[selectedModel];
  const matrix = selectedMetrics?.confusion_matrix;

  const modelSummary = useMemo(
    () => MODEL_NAMES.map((modelName) => ({
      modelName,
      status: status?.[modelName],
      metrics: metrics[modelName]?.metrics,
    })),
    [metrics, status],
  );

  async function handlePredict() {
    setPredicting(true);
    setPrediction(null);
    setPredictionError(null);
    try {
      const result = await api.predictModel(selectedModel, formFeatures);
      setPrediction(result);
    } catch (error) {
      setPredictionError(getErrorMessage(error));
    } finally {
      setPredicting(false);
    }
  }

  function handleFeatureChange(name: string, value: string) {
    setPrediction(null);
    setPredictionError(null);
    try {
      setFormFeatures((f) => ({ ...f, [name]: coerceFeatureValue(name, value) }));
    } catch (error) {
      setPredictionError(getErrorMessage(error));
    }
  }

  function applyCsvInput(csv: string) {
    try {
      setFormFeatures((f) => featuresFromCsv(csv, f));
      setPrediction(null);
      setPredictionError(null);
    } catch (error) {
      setPredictionError(getErrorMessage(error));
    }
  }

  async function handleCsvFile(file: File | undefined) {
    if (!file) return;
    const text = await file.text();
    setCsvInput(text);
    applyCsvInput(text);
  }

  /* ─── Status indicator helpers ──── */
  const sysStatus = loading ? "INITIALIZING" : loadError ? "BACKEND OFFLINE" : "SYSTEMS NOMINAL";
  const statusColor = loading
    ? "text-amber-400 bg-amber-500"
    : loadError
    ? "text-red-400 bg-red-500"
    : "text-emerald-400 bg-emerald-500";
  const [statusText, statusDot] = statusColor.split(" ");

  return (
    <main className="min-h-screen bg-[#020b14] cyber-grid px-4 py-6 text-slate-200 sm:px-6 sm:py-8">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-5">

        {/* ── Header ───────────────────────────────────────────── */}
        <header className="relative overflow-hidden rounded-sm border border-cyan-500/20 bg-slate-900 px-6 py-5 shadow-[0_0_40px_rgba(6,182,212,0.07)]">
          <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-500 to-transparent" />
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div className="space-y-1">
              <p className="font-mono text-[10px] font-semibold uppercase tracking-[0.32em] text-cyan-500">
                // INSIDER THREAT DETECTION
              </p>
              <h1 className="font-mono text-xl font-black tracking-tight text-slate-100 sm:text-2xl">
                THREAT INTELLIGENCE CENTER
              </h1>
              <p className="font-mono text-[10px] text-slate-600 uppercase tracking-widest">
                SYS :: {API_BASE_URL}
              </p>
            </div>
            <div className="flex items-center gap-2.5 self-start rounded-sm border border-slate-700 bg-slate-800/60 px-3 py-2 font-mono text-xs md:self-auto">
              <span className={`blink h-2 w-2 rounded-full ${statusDot}`} />
              <span className={statusText}>{sysStatus}</span>
            </div>
          </div>
        </header>

        {/* ── Load error ───────────────────────────────────────── */}
        {loadError ? (
          <section className="rounded-sm border border-red-500/30 bg-red-950/30 px-4 py-3 font-mono text-xs text-red-400">
            <span className="font-bold">[ERR]</span> {loadError}
          </section>
        ) : null}

        {/* ── Model selection grid ─────────────────────────────── */}
        <section className="grid gap-4 lg:grid-cols-3">
          {modelSummary.map(({ modelName, status: modelStatus, metrics: modelMetrics }) => {
            const isSelected = selectedModel === modelName;
            const isReady = modelStatus?.model_exists && modelStatus.metrics_exists;
            return (
              <button
                key={modelName}
                type="button"
                onClick={() => setSelectedModel(modelName)}
                className={`relative overflow-hidden rounded-sm border p-5 text-left transition-all ${
                  isSelected
                    ? "border-cyan-500/50 bg-slate-800/70 shadow-[0_0_24px_rgba(6,182,212,0.12)]"
                    : "border-slate-700/40 bg-slate-900 hover:border-slate-600/60 hover:bg-slate-800/40"
                }`}
              >
                {/* Selected left stripe */}
                {isSelected && (
                  <div className="absolute inset-y-0 left-0 w-[2px] bg-cyan-400" />
                )}
                {/* Top accent */}
                {isSelected && (
                  <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-500/60 to-transparent" />
                )}

                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="font-mono text-[9px] uppercase tracking-[0.3em] text-slate-500">
                      Algorithm Node
                    </p>
                    <h2 className="mt-1 font-mono text-base font-bold text-slate-100">
                      {MODEL_LABELS[modelName]}
                    </h2>
                  </div>
                  <span
                    className={`mt-1 rounded-sm px-2 py-1 font-mono text-[9px] font-bold uppercase tracking-widest ${
                      isReady
                        ? "border border-emerald-500/30 bg-emerald-950/50 text-emerald-400"
                        : "border border-amber-500/30 bg-amber-950/50 text-amber-400"
                    }`}
                  >
                    {isReady ? "READY" : "OFFLINE"}
                  </span>
                </div>

                <dl className="mt-4 grid grid-cols-2 gap-2.5 text-sm">
                  {(
                    [
                      { label: "ACC", value: formatPercent(modelMetrics?.accuracy) },
                      { label: "REC", value: formatPercent(modelMetrics?.recall) },
                      { label: "PRE", value: formatPercent(modelMetrics?.precision) },
                      { label: "F1", value: formatPercent(modelMetrics?.f1) },
                    ] as const
                  ).map(({ label, value }) => (
                    <div key={label} className="rounded-sm bg-slate-800/50 px-3 py-2.5">
                      <dt className="font-mono text-[9px] uppercase tracking-widest text-slate-500">
                        {label}
                      </dt>
                      <dd className="mt-1 font-mono text-lg font-bold text-slate-100">
                        {value}
                      </dd>
                    </div>
                  ))}
                </dl>
              </button>
            );
          })}
        </section>

        {/* ── Details + Prediction ─────────────────────────────── */}
        <section className="grid gap-4 lg:grid-cols-[0.95fr_1.05fr]">

          {/* ── Left: Model Details ──────────────────────────── */}
          <article className="rounded-sm border border-slate-700/40 bg-slate-900 shadow-[0_2px_20px_rgba(0,0,0,0.4)]">
            <div className="border-b border-slate-700/40 px-5 py-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="font-mono text-[9px] uppercase tracking-[0.28em] text-cyan-500/70">
                    // PERFORMANCE REPORT
                  </p>
                  <h2 className="mt-0.5 font-mono text-sm font-bold text-slate-100">
                    {MODEL_LABELS[selectedModel]}
                  </h2>
                </div>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value as ModelName)}
                  className="rounded-sm border border-slate-600 bg-slate-800 px-2.5 py-1.5 font-mono text-xs text-slate-300 outline-none focus:border-cyan-500/50"
                >
                  {MODEL_NAMES.map((n) => (
                    <option key={n} value={n}>{MODEL_LABELS[n]}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="p-5 space-y-5">
              <dl className="grid grid-cols-2 gap-3 text-sm">
                {(
                  [
                    { label: "FALSE POSITIVES", value: formatNumber(selectedMetrics?.false_positives) },
                    { label: "FALSE NEGATIVES", value: formatNumber(selectedMetrics?.false_negatives) },
                    { label: "ROC-AUC", value: formatPercent(selectedMetrics?.roc_auc) },
                    { label: "PR-AUC", value: formatPercent(selectedMetrics?.pr_auc) },
                  ] as const
                ).map(({ label, value }) => (
                  <div key={label} className="rounded-sm border border-slate-700/30 bg-slate-800/50 p-3">
                    <dt className="font-mono text-[9px] uppercase tracking-widest text-slate-500">
                      {label}
                    </dt>
                    <dd className="mt-1.5 font-mono text-2xl font-black text-slate-100">
                      {value}
                    </dd>
                  </div>
                ))}
              </dl>

              {/* Confusion Matrix */}
              <div>
                <p className="font-mono text-[9px] uppercase tracking-[0.28em] text-slate-500 mb-3">
                  // DETECTION MATRIX
                </p>
                <div className="grid grid-cols-2 overflow-hidden rounded-sm border border-slate-700/40 text-center text-sm">
                  <div className="border-b border-r border-slate-700/40 bg-emerald-950/30 p-4">
                    <p className="font-mono text-[9px] uppercase tracking-widest text-emerald-500/70">True Negatives</p>
                    <p className="mt-1.5 font-mono text-xl font-black text-emerald-400">
                      {formatNumber(matrix?.[0]?.[0])}
                    </p>
                  </div>
                  <div className="border-b border-slate-700/40 bg-amber-950/20 p-4">
                    <p className="font-mono text-[9px] uppercase tracking-widest text-amber-500/70">False Positives</p>
                    <p className="mt-1.5 font-mono text-xl font-black text-amber-400">
                      {formatNumber(matrix?.[0]?.[1])}
                    </p>
                  </div>
                  <div className="border-r border-slate-700/40 bg-red-950/30 p-4">
                    <p className="font-mono text-[9px] uppercase tracking-widest text-red-500/70">False Negatives</p>
                    <p className="mt-1.5 font-mono text-xl font-black text-red-400">
                      {formatNumber(matrix?.[1]?.[0])}
                    </p>
                  </div>
                  <div className="bg-cyan-950/20 p-4">
                    <p className="font-mono text-[9px] uppercase tracking-widest text-cyan-500/70">True Positives</p>
                    <p className="mt-1.5 font-mono text-xl font-black text-cyan-400">
                      {formatNumber(matrix?.[1]?.[1])}
                    </p>
                  </div>
                </div>
              </div>

              {/* Artifact paths */}
              <div className="space-y-1 rounded-sm border border-slate-700/30 bg-slate-800/30 px-3 py-2.5 font-mono text-[10px] text-slate-600">
                <p>MODEL :: {selectedStatus?.model_path ?? "n/a"}</p>
                <p>METRICS :: {selectedStatus?.metrics_path ?? "n/a"}</p>
              </div>
            </div>
          </article>

          {/* ── Right: Prediction Panel ──────────────────────── */}
          <article className="rounded-sm border border-slate-700/40 bg-slate-900 shadow-[0_2px_20px_rgba(0,0,0,0.4)]">
            <div className="border-b border-slate-700/40 px-5 py-4">
              <p className="font-mono text-[9px] uppercase tracking-[0.28em] text-cyan-500/70">
                // BEHAVIORAL THREAT ANALYZER
              </p>
              <h2 className="mt-0.5 font-mono text-sm font-bold text-slate-100">
                Prediction Test
              </h2>
              <p className="mt-0.5 font-mono text-[10px] text-slate-500">
                Enter feature values manually or load from CSV.
              </p>
            </div>

            <div className="p-5 space-y-5">
              {/* Feature inputs — spreadsheet */}
              <div className="overflow-hidden rounded-sm border border-slate-700/50 font-mono text-xs">
                {/* Header row */}
                <div className="grid grid-cols-[28px_1fr_1fr] border-b border-slate-600/60 bg-slate-800/80 select-none">
                  <div className="flex items-center justify-center border-r border-slate-700/50 py-2 text-[9px] uppercase tracking-widest text-slate-600">
                    #
                  </div>
                  <div className="border-r border-slate-700/50 px-3 py-2 text-[9px] uppercase tracking-widest text-slate-500">
                    Field
                  </div>
                  <div className="px-3 py-2 text-[9px] uppercase tracking-widest text-slate-500">
                    Value
                  </div>
                </div>

                {/* Data rows */}
                {FEATURE_FIELDS.map((field, i) => (
                  <div
                    key={field.name}
                    className={`grid grid-cols-[28px_1fr_1fr] border-b border-slate-700/30 last:border-b-0 ${
                      i % 2 === 0 ? "bg-slate-900/80" : "bg-slate-800/25"
                    }`}
                  >
                    {/* Row number */}
                    <div className="flex items-center justify-center border-r border-slate-700/30 text-[9px] text-slate-600 select-none">
                      {i + 1}
                    </div>
                    {/* Field name */}
                    <div className="flex items-center border-r border-slate-700/30 px-3 py-1.5 text-[10px] uppercase tracking-wider text-slate-400 select-none">
                      {field.label}
                    </div>
                    {/* Value cell */}
                    <div className="transition-colors focus-within:bg-cyan-950/40 focus-within:ring-1 focus-within:ring-inset focus-within:ring-cyan-500/50">
                      <input
                        type={field.type}
                        value={formFeatures[field.name]?.toString() ?? ""}
                        onChange={(e) => handleFeatureChange(field.name, e.target.value)}
                        className="w-full bg-transparent px-3 py-1.5 text-slate-100 outline-none"
                      />
                    </div>
                  </div>
                ))}
              </div>

              {/* CSV input */}
              <div className="rounded-sm border border-slate-700/40 bg-slate-800/30 p-4">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                  <div>
                    <p className="font-mono text-[9px] uppercase tracking-[0.28em] text-slate-400">
                      // CSV Input Terminal
                    </p>
                    <p className="mt-1 font-mono text-[10px] text-slate-600">
                      Use training column headers. First data row fills the form.
                    </p>
                  </div>
                  <input
                    type="file"
                    accept=".csv,text/csv"
                    onChange={(e) => handleCsvFile(e.target.files?.[0])}
                    className="font-mono text-xs text-slate-500 file:mr-3 file:rounded-sm file:border-0 file:bg-slate-700 file:px-3 file:py-1.5 file:font-mono file:text-xs file:font-bold file:text-slate-200 file:transition file:hover:bg-slate-600"
                  />
                </div>
                <textarea
                  value={csvInput}
                  onChange={(e) => setCsvInput(e.target.value)}
                  placeholder="employee_department,employee_campus,...&#10;IT,Campus A,..."
                  spellCheck={false}
                  className="mt-3 h-24 w-full resize-y rounded-sm border border-slate-700 bg-slate-900 p-3 font-mono text-[10px] leading-5 text-slate-400 outline-none transition focus:border-cyan-500/50 focus:shadow-[0_0_0_2px_rgba(6,182,212,0.1)]"
                />
                <button
                  type="button"
                  onClick={() => applyCsvInput(csvInput)}
                  className="mt-3 rounded-sm border border-slate-600 bg-slate-800 px-3 py-1.5 font-mono text-xs font-bold text-slate-300 transition hover:border-cyan-500/40 hover:bg-slate-700"
                >
                  LOAD FIRST ROW
                </button>
              </div>

              {/* Predict button */}
              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={handlePredict}
                  disabled={predicting}
                  className="rounded-sm bg-cyan-500/10 border border-cyan-500/40 px-5 py-2.5 font-mono text-sm font-black text-cyan-400 uppercase tracking-widest transition hover:bg-cyan-500/20 hover:border-cyan-400/60 hover:shadow-[0_0_16px_rgba(6,182,212,0.2)] disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {predicting ? "▶ ANALYZING..." : `▶ ANALYZE [${MODEL_LABELS[selectedModel]}]`}
                </button>
                {predictionError ? (
                  <p className="font-mono text-xs text-red-400">
                    <span className="font-bold">[ERR]</span> {predictionError}
                  </p>
                ) : null}
              </div>

              {/* Prediction result */}
              {prediction ? (
                <div className="space-y-4">
                  {/* Assessment banner */}
                  <div
                    className={`relative overflow-hidden rounded-sm border p-4 ${
                      prediction.prediction === 1
                        ? "border-red-500/40 bg-red-950/30 glow-red"
                        : "border-emerald-500/30 bg-emerald-950/20 glow-green"
                    }`}
                  >
                    <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-current to-transparent opacity-40" />
                    <p className="font-mono text-[9px] uppercase tracking-[0.28em] text-slate-500 mb-3">
                      // THREAT ASSESSMENT RESULT
                    </p>
                    <div className="grid gap-3 text-sm sm:grid-cols-3">
                      <div>
                        <p className="font-mono text-[9px] uppercase tracking-widest text-slate-500">
                          Prediction
                        </p>
                        <p className={`mt-1.5 font-mono text-2xl font-black ${
                          prediction.prediction === 1 ? "text-red-400" : "text-emerald-400"
                        }`}>
                          {prediction.prediction === 1 ? "MALICIOUS" : "CLEAR"}
                        </p>
                      </div>
                      <div>
                        <p className="font-mono text-[9px] uppercase tracking-widest text-slate-500">
                          Probability
                        </p>
                        <p className="mt-1.5 font-mono text-2xl font-black text-slate-100">
                          {formatPercent(prediction.probability)}
                        </p>
                      </div>
                      <div>
                        <p className="font-mono text-[9px] uppercase tracking-widest text-slate-500">
                          Threshold
                        </p>
                        <p className="mt-1.5 font-mono text-2xl font-black text-slate-100">
                          {formatPercent(prediction.decision_threshold)}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* SHAP Explanation */}
                  {prediction.explanation ? (
                    <div className="rounded-sm border border-slate-700/40 bg-slate-800/30 p-4 space-y-4">
                      <div className="flex flex-col gap-1 sm:flex-row sm:items-start sm:justify-between">
                        <div>
                          <p className="font-mono text-[9px] uppercase tracking-[0.28em] text-cyan-500/70">
                            // BEHAVIORAL INDICATORS
                          </p>
                          <p className="mt-0.5 font-mono text-[10px] text-slate-500">
                            SHAP contribution analysis — strongest drivers ranked.
                          </p>
                        </div>
                        <p className="font-mono text-[10px] text-slate-600">
                          BASE::{" "}
                          {prediction.explanation.base_value === null
                            ? "n/a"
                            : prediction.explanation.base_value.toFixed(4)}
                        </p>
                      </div>

                      {/* Summary */}
                      <div className="rounded-sm border border-cyan-500/20 bg-cyan-950/20 p-3 font-mono text-xs leading-6 text-cyan-300">
                        {shapSummary(prediction)}
                      </div>

                      {/* Top sentences */}
                      <ul className="space-y-2">
                        {prediction.explanation.top_contributions.slice(0, 5).map((c) => (
                          <li
                            key={`s-${c.feature}`}
                            className="rounded-sm border border-slate-700/30 bg-slate-800/40 px-3 py-2 font-mono text-xs leading-6 text-slate-400"
                          >
                            {shapSentence(c)}
                          </li>
                        ))}
                      </ul>

                      {/* Contribution details */}
                      <div>
                        <p className="font-mono text-[9px] uppercase tracking-[0.28em] text-slate-500 mb-3">
                          // CONTRIBUTION DETAILS
                        </p>
                        <div className="space-y-2">
                          {prediction.explanation.top_contributions.map((c) => (
                            <div
                              key={c.feature}
                              className={`grid gap-2 rounded-sm border px-3 py-2.5 text-sm sm:grid-cols-[1fr_auto] ${
                                c.direction === "increases_risk"
                                  ? "border-red-500/20 bg-red-950/20"
                                  : "border-emerald-500/15 bg-emerald-950/15"
                              }`}
                            >
                              <div>
                                <p className="font-mono text-xs font-semibold text-slate-200">
                                  {readableFeatureName(c.feature)}
                                </p>
                                <p className="font-mono text-[10px] text-slate-500">
                                  VAL :: {formatFeatureValue(c.value)}
                                </p>
                              </div>
                              <div className={`text-right font-mono font-bold ${
                                c.direction === "increases_risk"
                                  ? "text-red-400"
                                  : "text-emerald-400"
                              }`}>
                                <p className="text-sm">{formatShapValue(c.shap_value)}</p>
                                <p className="text-[9px] font-normal uppercase tracking-widest text-slate-500">
                                  {c.direction === "increases_risk" ? "↑ RISK" : "↓ RISK"}
                                </p>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>
          </article>
        </section>

      </div>
    </main>
  );
}
