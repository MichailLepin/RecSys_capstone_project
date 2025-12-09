// =====================================
// CONFIG
// =====================================
const TOTAL_CHUNKS = 17;
const ONNX_URL = "https://huggingface.co/iammik3e/recsys-minilm/resolve/main/model.onnx";
let recipes = [];
let session = null;
let tokenizer = null;

const loading = document.getElementById("loading");
const progress = document.getElementById("progress");

// =====================================
// IMPORT ONNX RUNTIME
// =====================================
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.js";

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
    const input = new Int32Array(maxLen);
    ids.forEach((v, i) => input[i] = v);
    // Заполняем padding нулями (если нужно)
    for (let i = ids.length; i < maxLen; i++) {
        input[i] = tokenizer.vocab["[PAD]"] ?? 0;
    }
    return new ort.Tensor("int32", input, [1, maxLen]);
}

// =====================================
// LOAD MODEL
// =====================================
async function loadOnnxModel() {
    console.log("Loading MiniLM ONNX…");

    session = await ort.InferenceSession.create(ONNX_URL, {
        executionProviders: ["wasm"]
    });

    console.log("MiniLM loaded ✔");
}

// =====================================
// LOAD RECIPE CHUNKS
// =====================================
async function loadChunks() {
    if (recipes.length > 0) return;

    let all = [];
    for (let i = 1; i <= TOTAL_CHUNKS; i++) {
        progress.textContent = `Loading recipes ${i}/${TOTAL_CHUNKS}…`;
        const chunk = await fetch(`chunks/part${i}.json`).then(r => r.json());
        all.push(...chunk);
    }

    recipes = all;
    progress.textContent = "";
    console.log("Loaded recipes:", recipes.length);
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
    loading.textContent = "Loading recipes…";
    await loadChunks();

    loading.textContent = "Loading tokenizer…";
    await loadTokenizer();

    loading.textContent = "Loading MiniLM model (first time is slow)…";
    await loadOnnxModel();

    loading.textContent = "Ready ✔";
}

init();

document.getElementById("searchBtn").onclick = recommend;
