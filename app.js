// ==========================================================
// SEMANTIC RECIPE RECOMMENDER — REAL S-BERT VERSION
// ==========================================================

// How many recipe chunks
const TOTAL_CHUNKS = 17;

let recipes = [];
let recipeEmbeddings = [];

let model; // SentenceTransformer

const loading = document.getElementById("loading");
const progress = document.getElementById("progress");

// ==========================================================
// 1. Load pretrained MiniLM model in browser
// ==========================================================
async function loadModel() {
    loading.textContent = "Loading Sentence-BERT model…";
    model = await window.sentenceTransformers.loadSentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2'
    );
    console.log("Model loaded ✓");
}

// ==========================================================
// 2. Load recipe chunks with embeddings
// ==========================================================
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

    console.log("Loaded", recipes.length, "recipes");

    // extract embeddings
    recipeEmbeddings = recipes.map(r => new Float32Array(r.embedding));

    console.log("Recipe embeddings ready ✓");
}

// ==========================================================
// 3. Cosine similarity
// ==========================================================
function cosineSimilarity(a, b) {
    let dot = 0, na = 0, nb = 0;

    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }

    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ==========================================================
// 4. Main Recommendation function
// ==========================================================
async function recommend() {
    const txt = document.getElementById("ingredientsInput").value.trim();
    if (!txt) return;

    loading.textContent = "Embedding user ingredients…";

    // Use Sentence-BERT to embed input
    const userEmb = await model.encode(txt);
    const userVec = new Float32Array(userEmb);

    loading.textContent = "Computing semantic similarity…";

    const scores = recipes.map((r, idx) => ({
        recipe: r,
        score: cosineSimilarity(userVec, recipeEmbeddings[idx])
    }));

    // Sort top 3
    const top = scores
        .sort((a, b) => b.score - a.score)
        .slice(0, 3);

    // Render
    document.getElementById("results").innerHTML = top.map(x => `
        <div class="recipe-card">
            <h3>${x.recipe.cuisine.toUpperCase()}</h3>
            <b>Score:</b> ${x.score.toFixed(4)}<br>
            <b>Ingredients:</b> ${x.recipe.ingredients.join(", ")}
        </div>
    `).join("");

    loading.textContent = "";
}

// ==========================================================
// 5. Init
// ==========================================================
async function init() {
    await loadModel();
    await loadChunks();
    
    loading.textContent = "Ready ✓";
}

init();

document.getElementById("searchBtn").onclick = recommend;
