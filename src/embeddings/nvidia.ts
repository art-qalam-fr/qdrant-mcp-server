import OpenAI from "openai";
import Bottleneck from "bottleneck";
import {
  EmbeddingProvider,
  EmbeddingResult,
  RateLimitConfig,
  ProviderConfig,
} from "./base.js";

interface NvidiaError {
  status?: number;
  code?: string;
  message?: string;
  headers?: Record<string, string>;
  response?: {
    headers?: Record<string, string>;
  };
}

export class NvidiaEmbeddings implements EmbeddingProvider {
  private client: OpenAI;
  private model: string;
  private dimensions: number;
  private limiter: Bottleneck;
  private retryAttempts: number;
  private retryDelayMs: number;
  private inputType: "passage" | "query";

  constructor(
    apiKey: string,
    model: string = "nvidia/nemotron-3-embed-1b",
    dimensions?: number,
    rateLimitConfig?: RateLimitConfig,
    baseUrl?: string,
    inputType: "passage" | "query" = "passage",
  ) {
    this.client = new OpenAI({
      apiKey,
      baseURL: baseUrl || "https://integrate.api.nvidia.com/v1",
    });
    this.model = model;
    this.inputType = inputType;

    // Default dimensions for NVIDIA NIM embedding models
    const defaultDimensions: Record<string, number> = {
      "nvidia/nemotron-3-embed-1b": 2048,
      "nvidia/llama-3.2-nv-embedqa-1b-v2": 2048,
      "nvidia/nv-embedqa-e5-v5": 1024,
      "nvidia/nv-embed-v1": 4096,
      "nvidia/nv-embedcode-7b-v1": 4096,
    };

    this.dimensions = dimensions || defaultDimensions[model] || 2048;

    // Rate limiting configuration
    const maxRequestsPerMinute = rateLimitConfig?.maxRequestsPerMinute || 1000;
    this.retryAttempts = rateLimitConfig?.retryAttempts || 3;
    this.retryDelayMs = rateLimitConfig?.retryDelayMs || 1000;

    // Initialize bottleneck limiter
    this.limiter = new Bottleneck({
      reservoir: maxRequestsPerMinute,
      reservoirRefreshAmount: maxRequestsPerMinute,
      reservoirRefreshInterval: 60 * 1000, // 1 minute
      maxConcurrent: 10,
      minTime: Math.floor((60 * 1000) / maxRequestsPerMinute),
    });
  }

  private async retryWithBackoff<T>(
    fn: () => Promise<T>,
    attempt: number = 0,
  ): Promise<T> {
    try {
      return await fn();
    } catch (error: unknown) {
      const apiError = error as NvidiaError;
      const isRateLimitError =
        apiError?.status === 429 ||
        apiError?.code === "rate_limit_exceeded" ||
        apiError?.message?.toLowerCase().includes("rate limit");

      if (isRateLimitError && attempt < this.retryAttempts) {
        // Check for Retry-After header
        const retryAfter =
          apiError?.response?.headers?.["retry-after"] ||
          apiError?.headers?.["retry-after"];
        let delayMs: number;

        if (retryAfter) {
          // Use Retry-After header if available (in seconds)
          const parsed = parseInt(retryAfter, 10);
          delayMs =
            !isNaN(parsed) && parsed > 0
              ? parsed * 1000
              : this.retryDelayMs * Math.pow(2, attempt);
        } else {
          // Exponential backoff: 1s, 2s, 4s, 8s...
          delayMs = this.retryDelayMs * Math.pow(2, attempt);
        }

        await new Promise((resolve) => setTimeout(resolve, delayMs));
        return this.retryWithBackoff(fn, attempt + 1);
      }

      // If not a rate limit error or max retries exceeded, throw
      if (isRateLimitError) {
        throw new Error(
          `NVIDIA NIM API rate limit exceeded after ${this.retryAttempts} retry attempts. Please try again later or reduce request frequency.`,
        );
      }

      throw error;
    }
  }

  async embed(text: string): Promise<EmbeddingResult> {
    return this.limiter.schedule(() =>
      this.retryWithBackoff(async () => {
        const response = await this.client.embeddings.create(
          {
            model: this.model,
            input: text,
            // NVIDIA NIM requires input_type ("passage" for indexing, "query" for search)
            input_type: this.inputType,
          } as Parameters<typeof this.client.embeddings.create>[0],
        );

        return {
          embedding: response.data[0].embedding,
          dimensions: this.dimensions,
        };
      }),
    );
  }

  async embedBatch(texts: string[]): Promise<EmbeddingResult[]> {
    return this.limiter.schedule(() =>
      this.retryWithBackoff(async () => {
        const response = await this.client.embeddings.create(
          {
            model: this.model,
            input: texts,
            input_type: this.inputType,
          } as Parameters<typeof this.client.embeddings.create>[0],
        );

        return response.data.map((item: OpenAI.Embedding) => ({
          embedding: item.embedding,
          dimensions: this.dimensions,
        }));
      }),
    );
  }

  getDimensions(): number {
    return this.dimensions;
  }

  getModel(): string {
    return this.model;
  }
}
