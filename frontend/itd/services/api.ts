import axios, { AxiosError } from "axios";

const DEFAULT_API_BASE_URL = "http://localhost:8000";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
  DEFAULT_API_BASE_URL;

export const MODEL_NAMES = ["xgboost", "lightgbm", "random_forest"] as const;

export type ModelName = (typeof MODEL_NAMES)[number];

export type PredictionFeatures = Record<string, string | number | boolean | null>;

export type PredictionRequest = {
  features: PredictionFeatures;
};

export type PredictionResponse = {
  model_name: string;
  model_path: string | null;
  probability: number;
  decision_threshold: number;
  prediction: 0 | 1;
};

export type ConfusionMatrix = [[number, number], [number, number]];

export type ModelMetrics = {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  confusion_matrix?: ConfusionMatrix;
  decision_threshold?: number;
  roc_auc?: number;
  pr_auc?: number;
  false_positives?: number;
  false_negatives?: number;
  true_positives?: number;
  true_negatives?: number;
  predicted_positive_rate?: number;
  [key: string]: unknown;
};

export type MetricsResponse = {
  model_name: ModelName;
  metrics_path: string;
  metrics: ModelMetrics;
};

export type ModelStatus = {
  model_path: string;
  model_exists: boolean;
  metrics_path: string;
  metrics_exists: boolean;
  loaded: boolean;
};

export type ModelsStatusResponse = Record<ModelName, ModelStatus>;

export type DefaultModelStatusResponse = {
  configured_model_path: string;
  exists: boolean;
  loaded: boolean;
  model_name: string | null;
};

type ApiErrorBody = {
  detail?: string;
};

export class ApiError extends Error {
  status: number;
  body: unknown;

  constructor(message: string, status: number, body: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

function toApiError(error: unknown): ApiError {
  if (error instanceof AxiosError) {
    const status = error.response?.status ?? 0;
    const body = error.response?.data;
    const detail =
      typeof body === "object" && body !== null
        ? (body as ApiErrorBody).detail
        : undefined;

    return new ApiError(
      detail ?? error.message ?? "Backend request failed",
      status,
      body,
    );
  }

  if (error instanceof Error) {
    return new ApiError(error.message, 0, null);
  }

  return new ApiError("Backend request failed", 0, null);
}

export function getDefaultModelStatus(): Promise<DefaultModelStatusResponse> {
  return apiClient
    .get<DefaultModelStatusResponse>("/model/status")
    .then((response) => response.data)
    .catch((error: unknown) => {
      throw toApiError(error);
    });
}

export function getModelsStatus(): Promise<ModelsStatusResponse> {
  return apiClient
    .get<ModelsStatusResponse>("/models/status")
    .then((response) => response.data)
    .catch((error: unknown) => {
      throw toApiError(error);
    });
}

export function getModelMetrics(modelName: ModelName): Promise<MetricsResponse> {
  return apiClient
    .get<MetricsResponse>(`/metrics/${modelName}`)
    .then((response) => response.data)
    .catch((error: unknown) => {
      throw toApiError(error);
    });
}

export function predictDefaultModel(
  features: PredictionFeatures,
): Promise<PredictionResponse> {
  return apiClient
    .post<PredictionResponse>("/predict", { features } satisfies PredictionRequest)
    .then((response) => response.data)
    .catch((error: unknown) => {
      throw toApiError(error);
    });
}

export function predictModel(
  modelName: ModelName,
  features: PredictionFeatures,
): Promise<PredictionResponse> {
  return apiClient
    .post<PredictionResponse>(`/predict/${modelName}`, {
      features,
    } satisfies PredictionRequest)
    .then((response) => response.data)
    .catch((error: unknown) => {
      throw toApiError(error);
    });
}

export const api = {
  getDefaultModelStatus,
  getModelsStatus,
  getModelMetrics,
  predictDefaultModel,
  predictModel,
};
