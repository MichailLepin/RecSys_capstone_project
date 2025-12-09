// =====================================
// CONFIG
// =====================================
const TOTAL_CHUNKS = 17;
const USE_SINGLE_FILE = true; // Использовать один файл вместо 17 чанков
const RECIPES_FILE = "recipes_all.json"; // Объединенный файл
const ONNX_URL = "https://huggingface.co/iammik3e/recsys-minilm/resolve/main/model.onnx";
let recipes = [];
let session = null;
let tokenizer = null;

const loading = document.getElementById("loading");
const progress = document.getElementById("progress");

// =====================================
// IMPORT ONNX RUNTIME
// =====================================
// ONNX Runtime будет загружен динамически
let ort = null;

// =====================================
// LOAD TOKENIZER FILES
// =====================================
async function loadTokenizer() {
    const vocabUrl = "model/vocab.txt";
    const text = await fetch(vocabUrl).then(r => r.text());
    const lines = text.split("\n");

    const vocab = {};
    lines.forEach((t, i) => vocab[t.trim()] = i);

    tokenizer = { vocab };

    console.log("Tokenizer loaded:", Object.keys(vocab).length, "tokens");
}

// Encode text into input_ids (VERY SIMPLIFIED tokenizer)
function tokenize(text) {
    text = text.toLowerCase().replace(/[^a-z0-9 ]+/g, " ");
    const tokens = text.split(" ").filter(t => t.length > 0);

    const ids = tokens.map(t => tokenizer.vocab[t] ?? tokenizer.vocab["[UNK]"]);
    // Добавляем [CLS] в начало (если есть в vocab)
    const clsId = tokenizer.vocab["[CLS]"] ?? tokenizer.vocab["[PAD]"] ?? 0;
    const paddedIds = [clsId, ...ids].slice(0, 128);
    
    return { ids: paddedIds, actualLength: Math.min(ids.length + 1, 128) };
}

// Convert ids→tensor for ONNX
function makeTensor(ids, maxLen = 128) {
    if (!ort) {
        throw new Error("ONNX Runtime not loaded yet");
    }
    const input = new Int32Array(maxLen);
    ids.forEach((v, i) => input[i] = v);
    // Заполняем padding нулями (если нужно)
    for (let i = ids.length; i < maxLen; i++) {
        input[i] = tokenizer.vocab["[PAD]"] ?? 0;
    }
    // Используем правильный API для создания тензора
    return new ort.Tensor("int32", input, [1, maxLen]);
}

// =====================================
// LOAD MODEL
// =====================================
async function loadOnnxModel() {
    console.log("Loading ONNX Runtime…");
    
    // Динамически загружаем ONNX Runtime
    if (!ort) {
        try {
            // Используем правильный путь для ONNX Runtime
            const ortModule = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js");
            ort = ortModule.default || ortModule;
            console.log("ONNX Runtime loaded ✔");
        } catch (error) {
            console.error("Failed to load ONNX Runtime via import:", error);
            // Попробуем альтернативный способ загрузки через script tag
            try {
                await new Promise((resolve, reject) => {
                    const script = document.createElement('script');
                    script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js';
                    script.type = 'text/javascript';
                    script.onload = () => {
                        ort = window.ort;
                        if (ort) {
                            console.log("ONNX Runtime loaded via script tag ✔");
                            resolve();
                        } else {
                            reject(new Error("ort not found on window"));
                        }
                    };
                    script.onerror = reject;
                    document.head.appendChild(script);
                });
            } catch (error2) {
                console.error("Failed to load ONNX Runtime via script:", error2);
                throw new Error("Failed to load ONNX Runtime. Please check your internet connection.");
            }
        }
    }

    if (!ort || !ort.InferenceSession) {
        throw new Error("ONNX Runtime not properly loaded. ort.InferenceSession is undefined.");
    }

    console.log("Loading MiniLM ONNX model…");

    try {
        session = await ort.InferenceSession.create(ONNX_URL, {
            executionProviders: ["wasm"]
        });
        console.log("MiniLM loaded ✔");
    } catch (error) {
        console.error("Failed to load ONNX model:", error);
        throw new Error("Failed to load ONNX model. Please check the model URL: " + ONNX_URL);
    }
}

// =====================================
// LOAD RECIPE CHUNKS (OPTIMIZED)
// =====================================
async function loadChunks() {
    if (recipes.length > 0) return;

    // Проверяем кэш в IndexedDB
    const cached = await loadFromCache();
    if (cached && cached.length > 0) {
        recipes = cached;
        console.log("Loaded recipes from cache:", recipes.length);
        return;
    }

    if (USE_SINGLE_FILE) {
        // Загрузка из одного объединенного файла (быстрее!)
        progress.textContent = `Loading recipes from single file…`;
        try {
            const response = await fetch(RECIPES_FILE);
            if (!response.ok) {
                throw new Error(`Failed to load ${RECIPES_FILE}, falling back to chunks`);
            }
            recipes = await response.json();
            progress.textContent = "";
            console.log("Loaded recipes from single file:", recipes.length);
            
            // Сохраняем в кэш
            await saveToCache(recipes);
            return;
        } catch (error) {
            console.warn("Single file not found, using chunks:", error);
            // Fallback к загрузке чанков
        }
    }

    // Fallback: Параллельная загрузка всех чанков одновременно
    progress.textContent = `Loading recipes (parallel)…`;
    const chunkPromises = [];
    
    for (let i = 1; i <= TOTAL_CHUNKS; i++) {
        chunkPromises.push(
            fetch(`chunks/part${i}.json`)
                .then(r => r.json())
                .then(data => {
                    progress.textContent = `Loading recipes ${i}/${TOTAL_CHUNKS}…`;
                    return data;
                })
        );
    }

    // Ждем загрузки всех чанков параллельно
    const chunks = await Promise.all(chunkPromises);
    
    // Объединяем все чанки
    let all = [];
    for (const chunk of chunks) {
        all.push(...chunk);
    }

    recipes = all;
    progress.textContent = "";
    console.log("Loaded recipes:", recipes.length);
    
    // Сохраняем в кэш для следующих раз
    await saveToCache(recipes);
}

// =====================================
// INDEXEDDB CACHE
// =====================================
let db = null;

async function initDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open("RecipeDB", 1);
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
            db = request.result;
            resolve(db);
        };
        
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains("recipes")) {
                db.createObjectStore("recipes", { keyPath: "id" });
            }
        };
    });
}

async function saveToCache(recipes) {
    try {
        if (!db) await initDB();
        
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(["recipes"], "readwrite");
            const store = transaction.objectStore("recipes");
            
            // Сохраняем как один большой объект
            const request = store.put({ id: "all", data: recipes, timestamp: Date.now() });
            
            request.onsuccess = () => {
                console.log("Recipes cached in IndexedDB");
                resolve();
            };
            
            request.onerror = () => {
                console.warn("Failed to cache recipes:", request.error);
                resolve(); // Не блокируем, если кэш не работает
            };
        });
    } catch (error) {
        console.warn("Failed to cache recipes:", error);
    }
}

async function loadFromCache() {
    try {
        if (!db) await initDB();
        
        const transaction = db.transaction(["recipes"], "readonly");
        const store = transaction.objectStore("recipes");
        const request = store.get("all");
        
        return new Promise((resolve, reject) => {
            request.onsuccess = () => {
                const result = request.result;
                if (result && result.data) {
                    // Проверяем, что кэш не старше 7 дней
                    const age = Date.now() - result.timestamp;
                    if (age < 7 * 24 * 60 * 60 * 1000) {
                        resolve(result.data);
                    } else {
                        resolve(null);
                    }
                } else {
                    resolve(null);
                }
            };
            request.onerror = () => resolve(null);
        });
    } catch (error) {
        console.warn("Failed to load from cache:", error);
        return null;
    }
}

// =====================================
// ENCODING USER INGREDIENTS
// =====================================
async function embed(text) {
    // ВАЖНО: Используем тот же формат, что и при создании эмбеддингов в Python
    // В Python: "Ingredients: " + ", ".join(ingredients)
    const formattedText = "Ingredients: " + text;
    
    const tokenized = tokenize(formattedText);
    const input_ids = makeTensor(tokenized.ids);
    
    // Создаем attention_mask (1 для реальных токенов, 0 для padding)
    if (!ort) {
        throw new Error("ONNX Runtime not loaded yet");
    }
    
    const attentionMask = new Int32Array(128);
    for (let i = 0; i < tokenized.actualLength; i++) {
        attentionMask[i] = 1;
    }
    for (let i = tokenized.actualLength; i < 128; i++) {
        attentionMask[i] = 0;
    }
    const attention_mask = new ort.Tensor("int32", attentionMask, [1, 128]);

    const outputs = await session.run({ 
        input_ids: input_ids,
        attention_mask: attention_mask 
    });
    
    // ПРОБЛЕМА БЫЛА ЗДЕСЬ: last_hidden_state имеет размерность [1, seq_len, 384]
    // Нужно применить mean pooling (усреднение по sequence_length, игнорируя padding)
    const lastHiddenState = outputs.last_hidden_state;
    
    // Получаем данные как массив
    const data = Array.from(lastHiddenState.data);
    const dims = lastHiddenState.dims; // [batch_size, seq_len, hidden_size]
    const seqLen = dims[1];
    const hiddenSize = dims[2];
    
    // Mean pooling с учетом attention_mask: усредняем только по реальным токенам
    const emb = new Array(hiddenSize).fill(0);
    const actualLength = tokenized.actualLength;
    
    for (let i = 0; i < hiddenSize; i++) {
        let sum = 0;
        for (let j = 0; j < actualLength; j++) {
            // Правильный индекс: [batch][seq][hidden] = seq * hidden_size + hidden
            const idx = j * hiddenSize + i;
            sum += data[idx];
        }
        emb[i] = sum / actualLength;
    }
    
    return emb;
}

// =====================================
// COSINE SIMILARITY
// =====================================
function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;

    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// =====================================
// MAIN SEARCH
// =====================================
async function recommend() {
    const txt = document.getElementById("ingredientsInput").value.trim();
    if (!txt) return;

    loading.textContent = "Encoding ingredients with MiniLM…";

    const userEmb = await embed(txt);

    loading.textContent = "Searching recipes…";

    const scores = recipes.map(r => ({
        recipe: r,
        score: cosine(userEmb, r.embedding)
    }));

    const top = scores.sort((a, b) => b.score - a.score).slice(0, 3);

    document.getElementById("results").innerHTML =
        top.map(x => `
        <div class="recipe-card">
            <h3>${x.recipe.cuisine.toUpperCase()}</h3>
            <b>Score:</b> ${x.score.toFixed(4)}<br>
            <b>Ingredients:</b> ${x.recipe.ingredients.join(", ")}<br><br>
            <i>This recipe matches your ingredients semantically, using MiniLM embedding similarity.</i>
        </div>
    `).join("");

    loading.textContent = "";
}

// =====================================
// INIT PIPELINE
// =====================================
async function init() {
    // Инициализируем БД для кэша
    await initDB();
    
    loading.textContent = "Initializing…";
    
    // Загружаем токенизатор и модель параллельно
    await Promise.all([
        (async () => {
            loading.textContent = "Loading tokenizer…";
            await loadTokenizer();
        })(),
        (async () => {
            loading.textContent = "Loading MiniLM model (first time is slow)…";
            await loadOnnxModel();
        })()
    ]);
    
    // Загружаем рецепты (может быть быстро из кэша)
    loading.textContent = "Loading recipes…";
    await loadChunks();

    loading.textContent = "Ready ✔";
}

init();

document.getElementById("searchBtn").onclick = recommend;
