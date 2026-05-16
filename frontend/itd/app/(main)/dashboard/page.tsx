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
  if (typeof value !== "number") {
    return "n/a";
  }
  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value: number | undefined) {
  if (typeof value !== "number") {
    return "n/a";
  }
  return value.toLocaleString();
}

function formatShapValue(value: number) {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(4)}`;
}

function formatFeatureValue(value: unknown) {
  if (value === null || value === undefined || value === "") {
    return "blank";
  }
  if (typeof value === "number") {
    return value.toLocaleString();
  }
  if (typeof value === "string" || typeof value === "boolean") {
    return value.toString();
  }
  return JSON.stringify(value);
}

function readableFeatureName(feature: string) {
  return FEATURE_LABELS[feature] ?? feature.replaceAll("_", " ");
}

function shapMagnitude(value: number) {
  const absoluteValue = Math.abs(value);
  if (absoluteValue >= 1) {
    return "strong";
  }
  if (absoluteValue >= 0.25) {
    return "moderate";
  }
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

  return `${feature} (${value}) had a ${shapMagnitude(
    contribution.shap_value,
  )} effect and ${direction}.`;
}

function shapSummary(prediction: PredictionResponse) {
  const topContributions = prediction.explanation?.top_contributions ?? [];
  const riskDrivers = topContributions.filter(
    (contribution) => contribution.direction === "increases_risk",
  );
  const protectiveDrivers = topContributions.filter(
    (contribution) => contribution.direction === "decreases_risk",
  );
  const strongestRisk = riskDrivers[0];
  const strongestProtective = protectiveDrivers[0];
  const predictedLabel =
    prediction.prediction === 1 ? "malicious" : "not malicious";

  if (!strongestRisk && !strongestProtective) {
    return `The model predicted ${predictedLabel}, but no SHAP drivers were returned.`;
  }

  if (strongestRisk && strongestProtective) {
    return `The model predicted ${predictedLabel}. The strongest risk driver was ${readableFeatureName(
      strongestRisk.feature,
    )}, while ${readableFeatureName(
      strongestProtective.feature,
    )} most reduced the risk score.`;
  }

  const strongestDriver = strongestRisk ?? strongestProtective;
  return `The model predicted ${predictedLabel}. The strongest driver was ${readableFeatureName(
    strongestDriver.feature,
  )}, which ${
    strongestDriver.direction === "increases_risk" ? "increased" : "reduced"
  } the risk score.`;
}

function getErrorMessage(error: unknown) {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Request failed";
}

function coerceFeatureValue(name: string, value: string) {
  const trimmed = value.trim();
  if (trimmed === "") {
    return null;
  }
  if (!NUMERIC_FEATURES.has(name)) {
    return trimmed;
  }

  const numberValue = Number(trimmed);
  if (Number.isNaN(numberValue)) {
    throw new Error(`${name} must be numeric.`);
  }
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

    if (character === '"') {
      inQuotes = !inQuotes;
      continue;
    }

    if (character === "," && !inQuotes) {
      row.push(cell);
      cell = "";
      continue;
    }

    if ((character === "\n" || character === "\r") && !inQuotes) {
      if (character === "\r" && nextCharacter === "\n") {
        index += 1;
      }
      row.push(cell);
      if (row.some((value) => value.trim() !== "")) {
        rows.push(row);
      }
      row = [];
      cell = "";
      continue;
    }

    cell += character;
  }

  row.push(cell);
  if (row.some((value) => value.trim() !== "")) {
    rows.push(row);
  }

  return rows;
}

function featuresFromCsv(csv: string, currentFeatures: PredictionFeatures) {
  const rows = parseCsvRows(csv);
  if (rows.length < 2) {
    throw new Error("CSV must include a header row and at least one data row.");
  }

  const headers = rows[0].map((header) => header.trim());
  const firstDataRow = rows[1];
  const nextFeatures = { ...currentFeatures };

  headers.forEach((header, index) => {
    if (!FEATURE_FIELDS.some((field) => field.name === header)) {
      return;
    }
    nextFeatures[header] = coerceFeatureValue(header, firstDataRow[index] ?? "");
  });

  return nextFeatures;
}

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
          Promise.all(MODEL_NAMES.map((modelName) => api.getModelMetrics(modelName))),
        ]);

        if (!isMounted) {
          return;
        }

        setStatus(modelStatus);
        setMetrics(
          metricResponses.reduce<Partial<Record<ModelName, MetricsResponse>>>(
            (currentMetrics, metricResponse) => ({
              ...currentMetrics,
              [metricResponse.model_name]: metricResponse,
            }),
            {},
          ),
        );
      } catch (error) {
        if (isMounted) {
          setLoadError(getErrorMessage(error));
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }

    loadDashboard();

    return () => {
      isMounted = false;
    };
  }, []);

  const selectedMetrics = metrics[selectedModel]?.metrics;
  const selectedStatus = status?.[selectedModel];
  const matrix = selectedMetrics?.confusion_matrix;

  const modelSummary = useMemo(
    () =>
      MODEL_NAMES.map((modelName) => ({
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
      setFormFeatures((currentFeatures) => ({
        ...currentFeatures,
        [name]: coerceFeatureValue(name, value),
      }));
    } catch (error) {
      setPredictionError(getErrorMessage(error));
    }
  }

  function applyCsvInput(csv: string) {
    try {
      setFormFeatures((currentFeatures) => featuresFromCsv(csv, currentFeatures));
      setPrediction(null);
      setPredictionError(null);
    } catch (error) {
      setPredictionError(getErrorMessage(error));
    }
  }

  async function handleCsvFile(file: File | undefined) {
    if (!file) {
      return;
    }

    const text = await file.text();
    setCsvInput(text);
    applyCsvInput(text);
  }

  return (
    <main className="min-h-screen bg-slate-100 px-6 py-8 text-slate-950">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6">
        <header className="flex flex-col gap-4 rounded-lg bg-slate-950 px-6 py-6 text-white shadow-sm md:flex-row md:items-end md:justify-between">
          <div className="space-y-2">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-sky-200">
              Insider threat detection
            </p>
            <h1 className="text-2xl font-semibold tracking-tight">
              Model monitoring dashboard
            </h1>
            <p className="max-w-2xl text-sm leading-6 text-slate-300">
              Backend: {API_BASE_URL}
            </p>
          </div>
          <div className="rounded-md border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-200">
            {loading ? "Loading" : loadError ? "Backend unavailable" : "Connected"}
          </div>
        </header>

        {loadError ? (
          <section className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">
            {loadError}
          </section>
        ) : null}

        <section className="grid gap-4 lg:grid-cols-3">
          {modelSummary.map(({ modelName, status: modelStatus, metrics: modelMetrics }) => (
            <button
              key={modelName}
              type="button"
              onClick={() => setSelectedModel(modelName)}
              className={`rounded-lg border bg-white p-5 text-left shadow-sm transition ${
                selectedModel === modelName
                  ? "border-sky-500 ring-2 ring-sky-100"
                  : "border-slate-200 hover:border-slate-300"
              }`}
            >
              <div className="flex items-center justify-between gap-3">
                <h2 className="text-lg font-semibold">{MODEL_LABELS[modelName]}</h2>
                <span
                  className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
                    modelStatus?.model_exists && modelStatus.metrics_exists
                      ? "bg-emerald-100 text-emerald-800"
                      : "bg-amber-100 text-amber-800"
                  }`}
                >
                  {modelStatus?.model_exists && modelStatus.metrics_exists
                    ? "Ready"
                    : "Missing files"}
                </span>
              </div>
              <dl className="mt-5 grid grid-cols-2 gap-4 text-sm">
                <div>
                  <dt className="text-slate-500">Accuracy</dt>
                  <dd className="mt-1 text-xl font-semibold">
                    {formatPercent(modelMetrics?.accuracy)}
                  </dd>
                </div>
                <div>
                  <dt className="text-slate-500">Recall</dt>
                  <dd className="mt-1 text-xl font-semibold">
                    {formatPercent(modelMetrics?.recall)}
                  </dd>
                </div>
                <div>
                  <dt className="text-slate-500">Precision</dt>
                  <dd className="mt-1 text-xl font-semibold">
                    {formatPercent(modelMetrics?.precision)}
                  </dd>
                </div>
                <div>
                  <dt className="text-slate-500">F1</dt>
                  <dd className="mt-1 text-xl font-semibold">
                    {formatPercent(modelMetrics?.f1)}
                  </dd>
                </div>
              </dl>
            </button>
          ))}
        </section>

        <section className="grid gap-4 lg:grid-cols-[0.95fr_1.05fr]">
          <article className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold">
                  {MODEL_LABELS[selectedModel]} details
                </h2>
                <p className="mt-1 text-sm text-slate-500">
                  Metrics from the saved backend report.
                </p>
              </div>
              <select
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value as ModelName)}
                className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm"
              >
                {MODEL_NAMES.map((modelName) => (
                  <option key={modelName} value={modelName}>
                    {MODEL_LABELS[modelName]}
                  </option>
                ))}
              </select>
            </div>

            <dl className="mt-6 grid grid-cols-2 gap-4 text-sm">
              <div className="rounded-md bg-slate-50 p-4">
                <dt className="text-slate-500">False positives</dt>
                <dd className="mt-1 text-2xl font-semibold">
                  {formatNumber(selectedMetrics?.false_positives)}
                </dd>
              </div>
              <div className="rounded-md bg-slate-50 p-4">
                <dt className="text-slate-500">False negatives</dt>
                <dd className="mt-1 text-2xl font-semibold">
                  {formatNumber(selectedMetrics?.false_negatives)}
                </dd>
              </div>
              <div className="rounded-md bg-slate-50 p-4">
                <dt className="text-slate-500">ROC-AUC</dt>
                <dd className="mt-1 text-2xl font-semibold">
                  {formatPercent(selectedMetrics?.roc_auc)}
                </dd>
              </div>
              <div className="rounded-md bg-slate-50 p-4">
                <dt className="text-slate-500">PR-AUC</dt>
                <dd className="mt-1 text-2xl font-semibold">
                  {formatPercent(selectedMetrics?.pr_auc)}
                </dd>
              </div>
            </dl>

            <div className="mt-6">
              <h3 className="text-sm font-semibold text-slate-700">
                Confusion matrix
              </h3>
              <div className="mt-3 grid grid-cols-2 overflow-hidden rounded-md border border-slate-200 text-center text-sm">
                <div className="border-b border-r border-slate-200 bg-emerald-50 p-4">
                  <p className="text-slate-500">True negatives</p>
                  <p className="mt-1 text-xl font-semibold">
                    {formatNumber(matrix?.[0]?.[0])}
                  </p>
                </div>
                <div className="border-b border-slate-200 bg-amber-50 p-4">
                  <p className="text-slate-500">False positives</p>
                  <p className="mt-1 text-xl font-semibold">
                    {formatNumber(matrix?.[0]?.[1])}
                  </p>
                </div>
                <div className="border-r border-slate-200 bg-rose-50 p-4">
                  <p className="text-slate-500">False negatives</p>
                  <p className="mt-1 text-xl font-semibold">
                    {formatNumber(matrix?.[1]?.[0])}
                  </p>
                </div>
                <div className="bg-sky-50 p-4">
                  <p className="text-slate-500">True positives</p>
                  <p className="mt-1 text-xl font-semibold">
                    {formatNumber(matrix?.[1]?.[1])}
                  </p>
                </div>
              </div>
            </div>

            <div className="mt-6 space-y-2 text-xs text-slate-500">
              <p>Model artifact: {selectedStatus?.model_path ?? "n/a"}</p>
              <p>Metrics report: {selectedStatus?.metrics_path ?? "n/a"}</p>
            </div>
          </article>

          <article className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <div>
              <h2 className="text-lg font-semibold">Prediction test</h2>
              <p className="mt-1 text-sm text-slate-500">
                Enter feature values manually or load the first row from a CSV.
              </p>
            </div>

            <div className="mt-4 grid gap-3 md:grid-cols-2">
              {FEATURE_FIELDS.map((field) => (
                <label key={field.name} className="text-sm">
                  <span className="font-medium text-slate-700">{field.label}</span>
                  <input
                    type={field.type}
                    value={formFeatures[field.name]?.toString() ?? ""}
                    onChange={(event) =>
                      handleFeatureChange(field.name, event.target.value)
                    }
                    className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm outline-none ring-sky-500 focus:ring-2"
                  />
                </label>
              ))}
            </div>

            <div className="mt-5 rounded-md border border-slate-200 bg-slate-50 p-4">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-slate-800">CSV input</h3>
                  <p className="mt-1 text-xs text-slate-500">
                    Use headers matching the training columns. The first data row
                    fills the form.
                  </p>
                </div>
                <input
                  type="file"
                  accept=".csv,text/csv"
                  onChange={(event) => handleCsvFile(event.target.files?.[0])}
                  className="text-sm text-slate-600 file:mr-3 file:rounded-md file:border-0 file:bg-slate-950 file:px-3 file:py-2 file:text-sm file:font-semibold file:text-white"
                />
              </div>

              <textarea
                value={csvInput}
                onChange={(event) => setCsvInput(event.target.value)}
                placeholder="employee_department,employee_campus,...&#10;IT,Campus A,..."
                spellCheck={false}
                className="mt-3 h-28 w-full resize-y rounded-md border border-slate-300 bg-white p-3 font-mono text-xs leading-5 outline-none ring-sky-500 focus:ring-2"
              />

              <button
                type="button"
                onClick={() => applyCsvInput(csvInput)}
                className="mt-3 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-800 transition hover:bg-slate-100"
              >
                Load first CSV row
              </button>
            </div>

            <div className="mt-4 flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={handlePredict}
                disabled={predicting}
                className="rounded-md bg-slate-950 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
              >
                {predicting ? "Predicting" : `Predict with ${MODEL_LABELS[selectedModel]}`}
              </button>
              {predictionError ? (
                <p className="text-sm text-rose-700">{predictionError}</p>
              ) : null}
            </div>

            {prediction ? (
              <div className="mt-5 space-y-4">
                <div className="grid gap-3 rounded-md border border-slate-200 bg-slate-50 p-4 text-sm sm:grid-cols-3">
                  <div>
                    <p className="text-slate-500">Prediction</p>
                    <p className="mt-1 text-2xl font-semibold">
                      {prediction.prediction === 1 ? "Malicious" : "Not malicious"}
                    </p>
                  </div>
                  <div>
                    <p className="text-slate-500">Probability</p>
                    <p className="mt-1 text-2xl font-semibold">
                      {formatPercent(prediction.probability)}
                    </p>
                  </div>
                  <div>
                    <p className="text-slate-500">Threshold</p>
                    <p className="mt-1 text-2xl font-semibold">
                      {formatPercent(prediction.decision_threshold)}
                    </p>
                  </div>
                </div>

                {prediction.explanation ? (
                  <div className="rounded-md border border-slate-200 bg-white p-4">
                    <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
                      <div>
                        <h3 className="text-sm font-semibold text-slate-800">
                          Explanation
                        </h3>
                        <p className="text-xs text-slate-500">
                          Plain-language summary of the strongest SHAP
                          contributions.
                        </p>
                      </div>
                      <p className="text-xs text-slate-500">
                        Base value:{" "}
                        {prediction.explanation.base_value === null
                          ? "n/a"
                          : prediction.explanation.base_value.toFixed(4)}
                      </p>
                    </div>

                    <div className="mt-4 rounded-md border border-sky-100 bg-sky-50 p-3 text-sm leading-6 text-sky-950">
                      {shapSummary(prediction)}
                    </div>

                    <ul className="mt-4 space-y-2">
                      {prediction.explanation.top_contributions
                        .slice(0, 5)
                        .map((contribution) => (
                          <li
                            key={`sentence-${contribution.feature}`}
                            className="rounded-md border border-slate-200 bg-white px-3 py-2 text-sm leading-6 text-slate-700"
                          >
                            {shapSentence(contribution)}
                          </li>
                        ))}
                    </ul>

                    <h4 className="mt-5 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                      Contribution details
                    </h4>
                    <div className="mt-4 space-y-2">
                      {prediction.explanation.top_contributions.map((contribution) => (
                        <div
                          key={contribution.feature}
                          className="grid gap-2 rounded-md bg-slate-50 p-3 text-sm sm:grid-cols-[1fr_auto]"
                        >
                          <div>
                            <p className="font-medium text-slate-800">
                              {readableFeatureName(contribution.feature)}
                            </p>
                            <p className="text-xs text-slate-500">
                              Value: {formatFeatureValue(contribution.value)}
                            </p>
                          </div>
                          <div
                            className={`text-right font-semibold ${
                              contribution.direction === "increases_risk"
                                ? "text-rose-700"
                                : "text-emerald-700"
                            }`}
                          >
                            {formatShapValue(contribution.shap_value)}
                            <p className="text-xs font-normal text-slate-500">
                              {contribution.direction === "increases_risk"
                                ? "increases risk"
                                : "decreases risk"}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>
            ) : null}
          </article>
        </section>
      </div>
    </main>
  );
}
