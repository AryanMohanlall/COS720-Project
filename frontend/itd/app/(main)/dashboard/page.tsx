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

function getErrorMessage(error: unknown) {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Request failed";
}

export default function Dashboard() {
  const [selectedModel, setSelectedModel] = useState<ModelName>("xgboost");
  const [status, setStatus] = useState<ModelsStatusResponse | null>(null);
  const [metrics, setMetrics] = useState<Partial<Record<ModelName, MetricsResponse>>>({});
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [featureJson, setFeatureJson] = useState(
    JSON.stringify(SAMPLE_FEATURES, null, 2),
  );
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
      const parsedFeatures = JSON.parse(featureJson) as PredictionFeatures;
      const result = await api.predictModel(selectedModel, parsedFeatures);
      setPrediction(result);
    } catch (error) {
      setPredictionError(getErrorMessage(error));
    } finally {
      setPredicting(false);
    }
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
                Send raw feature values to the selected model.
              </p>
            </div>

            <textarea
              value={featureJson}
              onChange={(event) => setFeatureJson(event.target.value)}
              spellCheck={false}
              className="mt-4 h-80 w-full resize-y rounded-md border border-slate-300 bg-slate-950 p-4 font-mono text-xs leading-5 text-slate-100 outline-none ring-sky-500 focus:ring-2"
            />

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
              <div className="mt-5 grid gap-3 rounded-md border border-slate-200 bg-slate-50 p-4 text-sm sm:grid-cols-3">
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
            ) : null}
          </article>
        </section>
      </div>
    </main>
  );
}
