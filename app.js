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

// Загружаем ONNX Runtime (правильный способ для ES модулей)
async function loadONNXRuntime() {
    // Пробуем использовать динамический импорт ES модулей
    try {
        const ortModule = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.wasm.min.js');
        // ONNX Runtime экспортирует объект с default или напрямую
        ort = ortModule.default || ortModule;
        
        // Проверяем структуру
        if (ort && ort.InferenceSession) {
            console.log("ONNX Runtime loaded via ES module ✔");
            return ort;
        }
        
        // Если структура неправильная, пробуем альтернативные пути
        if (ortModule.InferenceSession) {
            ort = ortModule;
            console.log("ONNX Runtime loaded via ES module (alternative path) ✔");
            return ort;
        }
    } catch (error) {
        console.warn("ES module import failed, trying script tag:", error);
    }
    
    // Fallback: загрузка через script tag
    if (window.ort) {
        ort = window.ort;
        return ort;
    }
    
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.wasm.min.js';
        script.async = true;
        script.type = 'text/javascript';
        
        script.onload = () => {
            setTimeout(() => {
                if (window.ort) {
                    ort = window.ort;
                    console.log("ONNX Runtime loaded via script tag ✔");
                    resolve(ort);
                } else {
                    reject(new Error('ONNX Runtime not found on window object'));
                }
            }, 200);
        };
        
        script.onerror = () => reject(new Error('Failed to load ONNX Runtime script'));
        document.head.appendChild(script);
    });
}

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
// ВАЖНО: Модель ожидает int64, а не int32!
function makeTensor(ids, maxLen = 128) {
    if (!ort) {
        throw new Error("ONNX Runtime not loaded yet");
    }
    // Используем BigInt64Array для int64 (модель ожидает int64)
    const input = new BigInt64Array(maxLen);
    ids.forEach((v, i) => input[i] = BigInt(v));
    // Заполняем padding нулями (если нужно)
    for (let i = ids.length; i < maxLen; i++) {
        input[i] = BigInt(tokenizer.vocab["[PAD]"] ?? 0);
    }
    // Используем правильный API для создания тензора с типом int64
    return new ort.Tensor('int64', input, [1, maxLen]);
}

// =====================================
// LOAD MODEL
// =====================================
async function loadOnnxModel() {
    loading.textContent = "Loading ONNX Runtime…";
    console.log("Loading ONNX Runtime…");
    
    // Загружаем ONNX Runtime
    if (!ort) {
        try {
            ort = await loadONNXRuntime();
            console.log("ONNX Runtime loaded ✔");
        } catch (error) {
            console.error("Failed to load ONNX Runtime:", error);
            loading.textContent = "Error: Failed to load ONNX Runtime";
            throw new Error("Failed to load ONNX Runtime. Please check your internet connection.");
        }
    }

    // Проверяем структуру ort объекта
    console.log("ort object structure:", {
        ort: !!ort,
        InferenceSession: !!ort?.InferenceSession,
        Tensor: typeof ort?.Tensor,
        keys: ort ? Object.keys(ort).slice(0, 10) : []
    });
    
    if (!ort) {
        loading.textContent = "Error: ONNX Runtime not loaded";
        throw new Error("ONNX Runtime not loaded");
    }
    
    // Проверяем разные возможные пути к InferenceSession
    if (!ort.InferenceSession) {
        // Возможно, нужно использовать ort.default.InferenceSession
        if (ort.default && ort.default.InferenceSession) {
            ort = ort.default;
        } else {
            // ort.InferenceSession отсутствует или falsy (null, undefined, false, 0, etc.)
            console.error("ort object:", ort);
            loading.textContent = "Error: ONNX Runtime structure incorrect";
            const availableKeys = ort ? Object.keys(ort).join(", ") : "none";
            throw new Error(`ONNX Runtime not properly loaded. ort.InferenceSession is ${ort.InferenceSession === undefined ? 'undefined' : 'falsy'}. Available keys: ${availableKeys}`);
        }
    }
    
    // Финальная проверка после возможного присваивания ort.default
    if (!ort.InferenceSession) {
        loading.textContent = "Error: ONNX Runtime InferenceSession not available";
        throw new Error("ONNX Runtime InferenceSession is not available after all checks");
    }

    loading.textContent = "Loading MiniLM model (this may take a moment)…";
    console.log("Loading MiniLM ONNX model…");

    try {
        session = await ort.InferenceSession.create(ONNX_URL, {
            executionProviders: ["wasm"]
        });
        console.log("MiniLM loaded ✔");
    } catch (error) {
        console.error("Failed to load ONNX model:", error);
        loading.textContent = "Error: Failed to load model";
        throw new Error("Failed to load ONNX model. Please check the model URL: " + ONNX_URL);
    }
}

// =====================================
// LOAD RECIPE CHUNKS (OPTIMIZED)
// =====================================
async function loadChunks() {
    if (recipes.length > 0) return;

    // Проверяем кэш в IndexedDB
    progress.textContent = "Checking cache…";
    const cached = await loadFromCache();
    if (cached && cached.length > 0) {
        recipes = cached;
        progress.textContent = "";
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
    progress.textContent = `Loading recipes chunks (0/${TOTAL_CHUNKS})…`;
    const chunkPromises = [];
    let loadedCount = 0;
    
    const updateProgress = () => {
        loadedCount++;
        progress.textContent = `Loading recipes chunks (${loadedCount}/${TOTAL_CHUNKS})…`;
    };
    
    for (let i = 1; i <= TOTAL_CHUNKS; i++) {
        chunkPromises.push(
            fetch(`chunks/part${i}.json`)
                .then(r => r.json())
                .then(data => {
                    updateProgress();
                    return data;
                })
                .catch(error => {
                    console.error(`Failed to load chunk ${i}:`, error);
                    updateProgress();
                    return []; // Возвращаем пустой массив при ошибке
                })
        );
    }

    // Ждем загрузки всех чанков параллельно
    const chunks = await Promise.all(chunkPromises);
    
    // Объединяем все чанки
    let all = [];
    for (const chunk of chunks) {
        if (chunk && chunk.length > 0) {
            all.push(...chunk);
        }
    }

    recipes = all;
    progress.textContent = "";
    console.log("Loaded recipes:", recipes.length);
    
    if (recipes.length === 0) {
        throw new Error("No recipes loaded. Please check your internet connection.");
    }
    
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
    
    // ВАЖНО: Модель ожидает int64 для всех входов!
    // Создаем attention_mask (1 для реальных токенов, 0 для padding)
    const attentionMask = new BigInt64Array(128);
    for (let i = 0; i < tokenized.actualLength; i++) {
        attentionMask[i] = BigInt(1);
    }
    for (let i = tokenized.actualLength; i < 128; i++) {
        attentionMask[i] = BigInt(0);
    }
    const attention_mask = new ort.Tensor("int64", attentionMask, [1, 128]);

    // Создаем token_type_ids (для одной последовательности все нули)
    const tokenTypeIds = new BigInt64Array(128);
    tokenTypeIds.fill(BigInt(0));
    const token_type_ids = new ort.Tensor("int64", tokenTypeIds, [1, 128]);

    // Проверяем, какие входы ожидает модель
    const inputNames = session.inputNames || [];
    console.log("Model input names:", inputNames);

    // Формируем входные данные - модель ожидает все три параметра
    const inputs = {
        input_ids: input_ids,
        attention_mask: attention_mask,
        token_type_ids: token_type_ids
    };

    try {
        const outputs = await session.run(inputs);
        return processModelOutput(outputs, tokenized.actualLength);
    } catch (error) {
        console.error("Model inference error:", error);
        console.error("Input shapes:", {
            input_ids: input_ids.dims,
            attention_mask: attention_mask.dims,
            token_type_ids: token_type_ids.dims
        });
        console.error("Input types:", {
            input_ids: input_ids.type,
            attention_mask: attention_mask.type,
            token_type_ids: token_type_ids.type
        });
        throw error;
    }
}

// Обработка выхода модели и применение mean pooling
function processModelOutput(outputs, actualLength) {
    // ПРОБЛЕМА БЫЛА ЗДЕСЬ: last_hidden_state имеет размерность [1, seq_len, 384]
    // Нужно применить mean pooling (усреднение по sequence_length, игнорируя padding)
    
    // Проверяем доступные выходы модели
    const outputNames = Object.keys(outputs);
    console.log("Model output names:", outputNames);
    
    // Ищем last_hidden_state в выходах (может быть под разными именами)
    let lastHiddenState = outputs.last_hidden_state || 
                         outputs['last_hidden_state'] ||
                         outputs[outputNames[0]]; // Используем первый выход, если не найдено
    
    if (!lastHiddenState) {
        throw new Error(`Could not find last_hidden_state in model outputs. Available outputs: ${outputNames.join(", ")}`);
    }
    
    // Получаем данные как массив
    const data = Array.from(lastHiddenState.data);
    const dims = lastHiddenState.dims; // [batch_size, seq_len, hidden_size]
    
    if (!dims || dims.length !== 3) {
        throw new Error(`Unexpected output shape. Expected [batch, seq_len, hidden_size], got: ${dims}`);
    }
    
    const seqLen = dims[1];
    const hiddenSize = dims[2];
    
    // Mean pooling с учетом attention_mask: усредняем только по реальным токенам
    const emb = new Array(hiddenSize).fill(0);
    
    for (let i = 0; i < hiddenSize; i++) {
        let sum = 0;
        for (let j = 0; j < actualLength; j++) {
            // Правильный индекс: [batch][seq][hidden] = seq * hidden_size + hidden
            const idx = j * hiddenSize + i;
            if (idx >= data.length) {
                console.warn(`Index ${idx} out of bounds for data length ${data.length}`);
                break;
            }
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
    if (!txt) {
        loading.textContent = "Please enter ingredients";
        return;
    }

    try {
        if (!session) {
            loading.textContent = "Model not loaded yet, please wait...";
            return;
        }

        if (recipes.length === 0) {
            loading.textContent = "Recipes not loaded yet, please wait...";
            return;
        }

        loading.textContent = "Encoding ingredients…";
        progress.textContent = "";

        const userEmb = await embed(txt);

        loading.textContent = "Searching recipes…";
        progress.textContent = `Comparing with ${recipes.length} recipes…`;

        const scores = recipes.map(r => ({
            recipe: r,
            score: cosine(userEmb, r.embedding)
        }));

        loading.textContent = "Sorting results…";
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
        progress.textContent = "";
    } catch (error) {
        console.error("Search error:", error);
        loading.textContent = `Error: ${error.message}`;
        progress.textContent = "";
    }
}

// =====================================
// INIT PIPELINE
// =====================================
async function init() {
    try {
        // Инициализируем БД для кэша
        loading.textContent = "Initializing…";
        await initDB();
        
        // Загружаем компоненты последовательно для лучшей индикации прогресса
        loading.textContent = "Loading tokenizer…";
        await loadTokenizer();
        console.log("Tokenizer loaded ✔");
        
        loading.textContent = "Loading ONNX Runtime…";
        await loadOnnxModel();
        console.log("Model loaded ✔");
        
        // Загружаем рецепты (может быть быстро из кэша)
        loading.textContent = "Loading recipes…";
        await loadChunks();
        console.log("Recipes loaded ✔");

        loading.textContent = "Ready ✔";
        progress.textContent = "";
    } catch (error) {
        console.error("Initialization error:", error);
        loading.textContent = `Error: ${error.message}`;
        progress.textContent = "Please refresh the page";
    }
}

init();

document.getElementById("searchBtn").onclick = recommend;
