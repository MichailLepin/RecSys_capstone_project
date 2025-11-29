// ======================================================
// IMPORT TRANSFORMERS.JS
// ======================================================
import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.1";

const loadingDiv = document.getElementById("loading");
const progressDiv = document.getElementById("progress");

// Модель SentenceTransformer, с которой совпадают твои эмбеддинги
const MODEL_ID = "Xenova/all-MiniLM-L6-v2";

// Количество JSON-чанков (исправь под своё число)
const TOTAL_CHUNKS = 17;

let embedder;
let modelReady = false;
let recipes = [];



// ======================================================
// LOAD MODEL
// ======================================================
async function initModel() {
    loadingDiv.textContent = "Loading embedding model… (~10–20 sec)";

    embedder = await pipeline(
        "feature-extraction",
        MODEL_ID
    );

    modelReady = true;
    loadingDiv.textContent = "Model loaded ✔";
}

initModel();



// ======================================================
// SAFE COSINE SIMILARITY
// ======================================================
function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;

    for (let i = 0; i < a.length; i++) {
        const x = Number.isFinite(a[i]) ? a[i] : 0;
        const y = Number.isFinite(b[i]) ? b[i] : 0;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }

    if (na === 0 || nb === 0) return 0;

    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}



// ======================================================
// LOAD CHUNKS (WITH NAN-CLEANING)
// ======================================================
async function loadChunks() {
    if (recipes.length > 0) return recipes;

    let all = [];

    for (let i = 1; i <= TOTAL_CHUNKS; i++) {
        progressDiv.textContent = `Loading chunk ${i}/${TOTAL_CHUNKS}…`;

        const chunk = await fetch(`chunks/part${i}.json`).then(r => r.json());

        // Clean NaN from embeddings
        chunk.forEach(r => {
            r.embedding = r.embedding.map(v =>
                Number.isFinite(v) ? v : 0
            );
        });

        all.push(...chunk);
    }

    progressDiv.textContent = "";
    recipes = all;

    console.log("Loaded recipes:", recipes.length);
    return recipes;
}



// ======================================================
// EMBED USER INPUT (WITH NAN-CLEANING)
// ======================================================
async function embedText(text) {
    const output = await embedder(text, {
        pooling: "mean",
        normalize: true,
    });

    let vec = Array.from(output.data[0]);

    // Clean NaN from user embedding
    vec = vec.map(v => Number.isFinite(v) ? v : 0);

    return vec;
}



// ======================================================
// MAIN RECOMMEND FUNCTION
// ======================================================
async function recommend() {
    if (!modelReady) {
        alert("Model is still loading…");
        return;
    }

    const text = document.getElementById("ingredientsInput").value.trim();
    if (!text) return;

    loadingDiv.textContent = "Embedding your ingredients…";

    const userEmbedding = await embedText(text);

    loadingDiv.textContent = "Loading recipes…";

    const recipesList = await loadChunks();

    loadingDiv.textContent = "Calculating similarity…";

    // Compute scores
    recipesList.forEach(r => {
        r.score = cosine(userEmbedding, r.embedding);
    });

    // Sort and take top 3
    const top = recipesList
        .sort((a, b) => b.score - a.score)
        .slice(0, 3);

    // Render UI
    document.getElementById("results").innerHTML = top.map(r => `
        <div class="recipe-card">
            <h3>${r.cuisine.toUpperCase()}</h3>
            <b>Score:</b> ${r.score.toFixed(4)}<br>
            <b>Ingredients:</b> ${r.ingredients.join(", ")}
        </div>
    `).join("");

    loadingDiv.textContent = "";
}



// ======================================================
// BUTTON HANDLER
// ======================================================
document.getElementById("searchBtn").onclick = recommend;
