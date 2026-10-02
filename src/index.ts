#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { QdrantClient } from "@qdrant/js-client-rest";
import { config } from "dotenv";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { appendFileSync, existsSync, mkdirSync } from "node:fs";

const __dirname = dirname(fileURLToPath(import.meta.url));
config({ path: join(__dirname, "../.env"), override: true });

// Logging
const logFile = join(__dirname, "../debug.log");
function logToFile(message: string): void {
  try {
    appendFileSync(logFile, `[${new Date().toISOString()}] ${message}\n`);
  } catch {
    // ignore
  }
}

logToFile("=== Qdrant Multi-Instance MCP Server starting ===");

// NOUVEAUTÉ: Map pour stocker plusieurs instances Qdrant
interface QdrantInstance {
  client: QdrantClient;
  url: string;
  dataDir?: string;
}

const instances: Map<string, QdrantInstance> = new Map();
let defaultInstanceAlias = "default";

// Créer ou récupérer une instance
function getOrCreateInstance(alias: string, customUrl?: string, customDataDir?: string): QdrantInstance {
  if (instances.has(alias)) {
    return instances.get(alias)!;
  }

  // Créer nouvelle instance
  const url = customUrl || process.env.QDRANT_URL || "http://localhost:6333";
  const dataDir = customDataDir || process.env.QDRANT_DATA_DIR;
  
  // Si on a un dataDir custom, on l'utilise pour créer une URL locale
  let finalUrl = url;
  if (customDataDir && !customUrl) {
    // En mode local avec dataDir spécifique, on utilise le dataDir comme identifiant
    // La vraie connexion Qdrant reste sur le même serveur, mais on tague les collections
    finalUrl = url;
  }

  const client = new QdrantClient({ url: finalUrl, checkCompatibility: false });
  const instance: QdrantInstance = { client, url: finalUrl, dataDir };
  instances.set(alias, instance);
  
  logToFile(`Created Qdrant instance '${alias}' at ${finalUrl}`);
  
  return instance;
}

function getInstance(alias?: string): QdrantInstance {
  const targetAlias = alias || defaultInstanceAlias;
  const instance = instances.get(targetAlias);
  if (!instance) {
    throw new Error(`Qdrant instance '${targetAlias}' not found. Use create_instance first.`);
  }
  return instance;
}

// Fonctions utilitaires pour Qdrant
async function createCollection(
  instance: QdrantInstance,
  name: string,
  vectorSize: number,
  distance: "Cosine" | "Euclid" | "Dot" = "Cosine",
  enableSparse: boolean = false
): Promise<void> {
  const config: any = {};
  if (enableSparse) {
    config.vectors = { dense: { size: vectorSize, distance } };
    config.sparse_vectors = { text: { modifier: "idf" } };
  } else {
    config.vectors = { size: vectorSize, distance };
  }
  await instance.client.createCollection(name, config);
}

async function collectionExists(instance: QdrantInstance, name: string): Promise<boolean> {
  try {
    await instance.client.getCollection(name);
    return true;
  } catch {
    return false;
  }
}

async function listCollections(instance: QdrantInstance): Promise<string[]> {
  const response = await instance.client.getCollections();
  return response.collections.map((c: any) => c.name);
}

async function getCollectionInfo(instance: QdrantInstance, name: string): Promise<any> {
  const info = await instance.client.getCollection(name);
  const vectorConfig = info.config.params.vectors;
  let size = 0;
  let distance: "Cosine" | "Euclid" | "Dot" = "Cosine";
  let hybridEnabled = false;

  if (info.config.params.sparse_vectors) {
    hybridEnabled = true;
  }

  if (typeof vectorConfig === "object" && vectorConfig !== null) {
    if ("size" in vectorConfig) {
      size = typeof vectorConfig.size === "number" ? vectorConfig.size : 0;
      distance = vectorConfig.distance as "Cosine" | "Euclid" | "Dot";
    } else if ("dense" in vectorConfig) {
      const denseConfig = (vectorConfig as any).dense;
      size = typeof denseConfig.size === "number" ? denseConfig.size : 0;
      distance = denseConfig.distance as "Cosine" | "Euclid" | "Dot";
    }
  }

  return {
    name,
    vectorSize: size,
    pointsCount: info.points_count || 0,
    distance,
    hybridEnabled,
  };
}

async function deleteCollection(instance: QdrantInstance, name: string): Promise<void> {
  await instance.client.deleteCollection(name);
}

async function addPoints(
  instance: QdrantInstance,
  collectionName: string,
  points: Array<{ id: string | number; vector: number[]; payload?: Record<string, any> }>
): Promise<void> {
  await instance.client.upsert(collectionName, {
    wait: true,
    points: points.map((p) => ({
      id: p.id,
      vector: p.vector,
      payload: p.payload,
    })),
  });
}

async function search(
  instance: QdrantInstance,
  collectionName: string,
  vector: number[],
  limit: number = 5,
  filter?: Record<string, any>
): Promise<any[]> {
  const results = await instance.client.search(collectionName, {
    vector,
    limit,
    filter,
  });
  return results.map((r: any) => ({
    id: r.id,
    score: r.score,
    payload: r.payload,
  }));
}

async function deletePoints(
  instance: QdrantInstance,
  collectionName: string,
  ids: (string | number)[]
): Promise<void> {
  await instance.client.delete(collectionName, {
    wait: true,
    points: ids,
  });
}

// ===================================================================
// MCP Server Setup
// ===================================================================

const server = new Server(
  { name: "qdrant-mcp-server-multi", version: "2.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "create_instance",
        description: "Create a new Qdrant instance with specific data directory or URL",
        inputSchema: {
          type: "object",
          properties: {
            alias: { type: "string", description: "Alias for this instance (default: 'default')" },
            url: { type: "string", description: "Qdrant URL (optional, uses env default)" },
            data_dir: { type: "string", description: "Local data directory for storage path tagging" },
          },
          required: ["alias"],
        },
      },
      {
        name: "list_instances",
        description: "List all configured Qdrant instances",
        inputSchema: { type: "object", properties: {} },
      },
      {
        name: "switch_instance",
        description: "Switch the default instance alias",
        inputSchema: {
          type: "object",
          properties: {
            alias: { type: "string", description: "Instance alias to set as default" },
          },
          required: ["alias"],
        },
      },
      {
        name: "create_collection",
        description: "Create a vector collection in specified instance",
        inputSchema: {
          type: "object",
          properties: {
            name: { type: "string", description: "Collection name" },
            vectorSize: { type: "number", description: "Vector dimensions" },
            distance: { type: "string", enum: ["Cosine", "Euclid", "Dot"], default: "Cosine" },
            enableHybrid: { type: "boolean", default: false },
            instance_alias: { type: "string", description: "Instance alias (default: current default)", default: "default" },
          },
          required: ["name", "vectorSize"],
        },
      },
      {
        name: "list_collections",
        description: "List collections in specified instance",
        inputSchema: {
          type: "object",
          properties: {
            instance_alias: { type: "string", default: "default" },
          },
        },
      },
      {
        name: "get_collection_info",
        description: "Get collection info from specified instance",
        inputSchema: {
          type: "object",
          properties: {
            name: { type: "string", description: "Collection name" },
            instance_alias: { type: "string", default: "default" },
          },
          required: ["name"],
        },
      },
      {
        name: "delete_collection",
        description: "Delete collection from specified instance",
        inputSchema: {
          type: "object",
          properties: {
            name: { type: "string", description: "Collection name" },
            instance_alias: { type: "string", default: "default" },
          },
          required: ["name"],
        },
      },
      {
        name: "add_documents",
        description: "Add documents to collection in specified instance",
        inputSchema: {
          type: "object",
          properties: {
            collection: { type: "string", description: "Collection name" },
            documents: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  id: { type: "string" },
                  vector: { type: "array", items: { type: "number" } },
                  payload: { type: "object" },
                },
                required: ["id", "vector"],
              },
            },
            instance_alias: { type: "string", default: "default" },
          },
          required: ["collection", "documents"],
        },
      },
      {
        name: "search",
        description: "Search collection in specified instance",
        inputSchema: {
          type: "object",
          properties: {
            collection: { type: "string" },
            vector: { type: "array", items: { type: "number" } },
            limit: { type: "number", default: 5 },
            filter: { type: "object" },
            instance_alias: { type: "string", default: "default" },
          },
          required: ["collection", "vector"],
        },
      },
      {
        name: "delete_documents",
        description: "Delete documents from collection in specified instance",
        inputSchema: {
          type: "object",
          properties: {
            collection: { type: "string" },
            ids: { type: "array", items: { type: "string" } },
            instance_alias: { type: "string", default: "default" },
          },
          required: ["collection", "ids"],
        },
      },
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "create_instance": {
        const { alias, url, data_dir } = args as { alias: string; url?: string; data_dir?: string };
        
        // Créer le dossier data_dir si spécifié
        if (data_dir) {
          const resolvedDir = resolve(data_dir);
          if (!existsSync(resolvedDir)) {
            mkdirSync(resolvedDir, { recursive: true });
          }
        }
        
        const instance = getOrCreateInstance(alias, url, data_dir);
        
        if (instances.size === 1) {
          defaultInstanceAlias = alias;
        }

        return {
          content: [
            {
              type: "text",
              text: `Created Qdrant instance '${alias}'\nURL: ${instance.url}\nDataDir: ${instance.dataDir || "none"}\nTotal instances: ${instances.size}`,
            },
          ],
        };
      }

      case "list_instances": {
        const list = [];
        for (const [alias, inst] of instances) {
          list.push(`- ${alias === defaultInstanceAlias ? "[DEFAULT] " : ""}${alias}: ${inst.url} ${inst.dataDir ? "(dir: " + inst.dataDir + ")" : ""}`);
        }
        return {
          content: [{ type: "text", text: `Instances (${instances.size}):\n${list.join("\n")}` }],
        };
      }

      case "switch_instance": {
        const { alias } = args as { alias: string };
        if (!instances.has(alias)) {
          throw new Error(`Instance '${alias}' not found`);
        }
        defaultInstanceAlias = alias;
        return {
          content: [{ type: "text", text: `Switched to instance: ${alias}` }],
        };
      }

      case "create_collection": {
        const { name, vectorSize, distance, enableHybrid, instance_alias } = args as {
          name: string;
          vectorSize: number;
          distance?: "Cosine" | "Euclid" | "Dot";
          enableHybrid?: boolean;
          instance_alias?: string;
        };
        const instance = getInstance(instance_alias);
        await createCollection(instance, name, vectorSize, distance, enableHybrid);
        return {
          content: [{ type: "text", text: `Created collection '${name}' in instance '${instance_alias || defaultInstanceAlias}'` }],
        };
      }

      case "list_collections": {
        const { instance_alias } = args as { instance_alias?: string };
        const instance = getInstance(instance_alias);
        const collections = await listCollections(instance);
        return {
          content: [{ type: "text", text: `Collections in '${instance_alias || defaultInstanceAlias}': ${collections.join(", ") || "none"}` }],
        };
      }

      case "get_collection_info": {
        const { name, instance_alias } = args as { name: string; instance_alias?: string };
        const instance = getInstance(instance_alias);
        const info = await getCollectionInfo(instance, name);
        return {
          content: [{ type: "text", text: `Collection: ${name}\nSize: ${info.vectorSize}\nPoints: ${info.pointsCount}\nDistance: ${info.distance}\nHybrid: ${info.hybridEnabled}` }],
        };
      }

      case "delete_collection": {
        const { name, instance_alias } = args as { name: string; instance_alias?: string };
        const instance = getInstance(instance_alias);
        await deleteCollection(instance, name);
        return {
          content: [{ type: "text", text: `Deleted collection '${name}' from instance '${instance_alias || defaultInstanceAlias}'` }],
        };
      }

      case "add_documents": {
        const { collection, documents, instance_alias } = args as {
          collection: string;
          documents: Array<{ id: string; vector: number[]; payload?: Record<string, any> }>;
          instance_alias?: string;
        };
        const instance = getInstance(instance_alias);
        await addPoints(instance, collection, documents);
        return {
          content: [{ type: "text", text: `Added ${documents.length} documents to '${collection}' in instance '${instance_alias || defaultInstanceAlias}'` }],
        };
      }

      case "search": {
        const { collection, vector, limit, filter, instance_alias } = args as {
          collection: string;
          vector: number[];
          limit?: number;
          filter?: Record<string, any>;
          instance_alias?: string;
        };
        const instance = getInstance(instance_alias);
        const results = await search(instance, collection, vector, limit, filter);
        return {
          content: [{ type: "text", text: `Search results (${results.length}):\n${JSON.stringify(results, null, 2)}` }],
        };
      }

      case "delete_documents": {
        const { collection, ids, instance_alias } = args as {
          collection: string;
          ids: string[];
          instance_alias?: string;
        };
        const instance = getInstance(instance_alias);
        await deletePoints(instance, collection, ids);
        return {
          content: [{ type: "text", text: `Deleted ${ids.length} documents from '${collection}' in instance '${instance_alias || defaultInstanceAlias}'` }],
        };
      }

      default:
        return { content: [{ type: "text", text: `Unknown tool: ${name}` }], isError: true };
    }
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    logToFile(`Error in ${name}: ${msg}`);
    return { content: [{ type: "text", text: `Error: ${msg}` }], isError: true };
  }
});

// ===================================================================
// Start Server
// ===================================================================

async function main() {
  logToFile("Starting Qdrant Multi-Instance MCP Server...");

  // Créer l'instance par défaut
  const defaultUrl = process.env.QDRANT_URL || "http://localhost:6333";
  const defaultDataDir = process.env.QDRANT_DATA_DIR;
  getOrCreateInstance("default", defaultUrl, defaultDataDir);

  const transport = new StdioServerTransport();
  await server.connect(transport);

  logToFile("Qdrant Multi-Instance MCP Server ready");
}

main().catch((error) => {
  logToFile(`Fatal error: ${error}`);
  process.exit(1);
});
