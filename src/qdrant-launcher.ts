import { spawn, ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export class QdrantLauncher {
  private qdrantProcess: ChildProcess | null = null;
  private readonly qdrantPath: string;
  private readonly storagePath: string;

  private async _checkUrlOk(url: string): Promise<boolean> {
    try {
      const response = await fetch(url);
      return response.ok;
    } catch {
      return false;
    }
  }

  constructor(qdrantPath?: string, storagePath?: string) {
    // Chemin par défaut vers l'exécutable Qdrant
    this.qdrantPath = qdrantPath || "<HEPHAISTOS_ROOT>\\qdrant\\qdrant.exe";

    // Chemin de stockage dédié pour éviter les conflits
    const __dirname = dirname(fileURLToPath(import.meta.url));
    this.storagePath = storagePath || join(__dirname, "../ide_storage");
  }

  /**
   * Vérifie si Qdrant est déjà en cours d'exécution
   */
  async isRunning(): Promise<boolean> {
    const base = "http://127.0.0.1:6333";
    const endpoints = ["/readyz", "/ready", "/", "/health"];
    for (const ep of endpoints) {
      if (await this._checkUrlOk(`${base}${ep}`)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Démarre Qdrant ou utilise l'instance existante
   */
  async start(): Promise<void> {
    // Vérifier si Qdrant est déjà en cours d'exécution
    if (await this.isRunning()) {
      // console.error("Qdrant is already running, using existing instance"); // Désactivé
      return;
    }

    // console.error("No existing Qdrant instance found, starting new one..."); // Désactivé

    // Vérifier si l'exécutable existe
    if (!existsSync(this.qdrantPath)) {
      throw new Error(`Qdrant executable not found at: ${this.qdrantPath}`);
    }

    // Créer le répertoire de stockage s'il n'existe pas
    const fs = await import("node:fs");
    if (!existsSync(this.storagePath)) {
      fs.mkdirSync(this.storagePath, { recursive: true });
    }

    return new Promise((resolve, reject) => {
      // console.error(`Starting Qdrant from: ${this.qdrantPath}`); // Désactivé

      // Préparer les variables d'environnement pour Qdrant
      const env = {
        ...process.env,
        QDRANT__STORAGE__STORAGE_PATH: this.storagePath,
        QDRANT__SERVICE__HTTP_PORT: "6333",
        QDRANT__LOG__LEVEL: "ERROR",
        QDRANT__TELEMETRY_DISABLED: "true"
      };

      // Lancer Qdrant avec configuration par variables d'environnement
      this.qdrantProcess = spawn(this.qdrantPath, [], {
        detached: false, // Garder le processus attaché pour le debug
        stdio: "pipe", // Capturer la sortie
        windowsHide: true, // Cacher la fenêtre en production
        cwd: dirname(this.qdrantPath), // Exécuter depuis le répertoire de Qdrant
        env: env // Variables d'environnement personnalisées
      });

      // Capturer la sortie pour diagnostiquer
      if (this.qdrantProcess.stdout) {
        this.qdrantProcess.stdout.on("data", (data) => {
          const output = data.toString().trim();
          if (output && !output.includes("Access web UI")) {
            // console.error(`Qdrant: ${output}`); // Désactivé
          }
        });
      }

      if (this.qdrantProcess.stderr) {
        this.qdrantProcess.stderr.on("data", (data) => {
          const output = data.toString().trim();
          if (output && !output.includes("WARN") && !output.includes("Config file not found")) {
            // console.error(`Qdrant ERROR: ${output}`); // Désactivé
          }
        });
      }

      // Détacher le processus pour qu'il continue en arrière-plan
      if (this.qdrantProcess.pid) {
        this.qdrantProcess.unref();
      }

      this.qdrantProcess.on("error", (error) => {
        reject(new Error(`Failed to start Qdrant: ${error.message}`));
      });

      this.qdrantProcess.on("spawn", () => {
        // console.error("Qdrant process spawned, waiting for readiness..."); // Désactivé
      });

      // Attendre que Qdrant soit prêt
      this.waitForReady()
        .then(() => {
          // console.error("Qdrant is ready and running"); // Désactivé
          resolve();
        })
        .catch(reject);
    });
  }

  /**
   * Attend que Qdrant soit prêt à accepter des connexions
   */
  private async waitForReady(maxAttempts = 60, delay = 1000): Promise<void> {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const base = "http://localhost:6333";
      const endpoints = ["/readyz", "/ready", "/", "/health"];

      for (const ep of endpoints) {
        try {
          const response = await fetch(`${base}${ep}`);
          if (response.ok) return;
        } catch {}
      }

      if (attempt % 5 === 0) {
        // Log simple pour suivre la progression sans polluer trop
        // console.error(`Waiting for Qdrant... attempt ${attempt + 1}/${maxAttempts}`);
      }

      await new Promise(resolve => setTimeout(resolve, delay));
    }

    throw new Error(`Qdrant failed to start within the expected time (${maxAttempts}s)`);
  }

  /**
   * Arrête le processus Qdrant si nous l'avons lancé
   */
  async stop(): Promise<void> {
    if (this.qdrantProcess && !this.qdrantProcess.killed) {
      this.qdrantProcess.kill();
      this.qdrantProcess = null;
      // console.error("Qdrant process stopped"); // Désactivé
    }
  }

  /**
   * Nettoie les ressources
   */
  async cleanup(): Promise<void> {
    await this.stop();
  }
}
