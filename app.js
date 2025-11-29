const MODEL_PATH = "model.onnx";     // MiniLM ONNX
const VOCAB_PATH = "vocab.json";     // tokenizer vocab
const TOTAL_CHUNKS = 17;

let tokenizer;
let session;
let recipes = [];

const loadingDiv = document.getElementById("loading");
const progressDiv = document.getElementById("progress");

// ------------------------------
// Load model + tokenizer
// ------------------------------
async function initModel() {
    loadingDiv.textContent = "Loading MiniLM model…";

    const vocab = await fetch(VOCAB_PATH).then(r => r.json());
    tokenizer = new BertTokenizer(vocab);

    session = await ort.InferenceSession.create(MODEL_PATH);

    loadingDiv.textContent = "Model ready ✔";
}
initModel();

// ------------------------------
// Cosine similarity
// ------------------------------
function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;

    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }

    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ------------------------------
// Load recipe chunks once
// ------------------------------
async function loadChunks() {
    if (recipes.length > 0) return recipes;

    let all = [];
    for (let i = 1; i <= TOTAL_CHUNKS; i++) {
        progressDiv.textContent = `Loading chunk ${i}/${TOTAL_CHUNKS}…`;

        const chunk = await fetch(`chunks/part${i}.json`).then(r => r.json());
        all.push(...chunk);
    }
    progressDiv.textContent = "";
    recipes = all;
    return recipes;
}

// ------------------------------
// Compute MiniLM embedding
// ------------------------------
async function embed(text) {
    const { ids, mask } = tokenizer.encode(text);

    const inputs = {
        input_ids: new ort.Tensor("int64", ids, [1, 128]),
        attention_mask: new ort.Tensor("int64", new BigInt64Array(mask.map(x=>BigInt(x))), [1, 128])
    };

    const output = await session.run(inputs);
    const last = output["last_hidden_state"].data;

    // Mean pooling
    const dim = 384;
    let vec = new Array(dim).fill(0);
    for (let i = 0; i < 128; i++) {
        for (let d = 0; d < dim; d++) {
            vec[d] += Number(last[i * dim + d]);
        }
    }
    return vec.map(v => v / 128);
}

// ------------------------------
// Main recommend function
// ------------------------------
async function recommend() {
    const text = document.getElementById("ingredientsInput").value.trim();
    if (!text) return;

    loadingDiv.textContent = "Embedding input…";
    const userVec = await embed(text);

    loadingDiv.textContent = "Loading recipes…";
    const data = await loadChunks();

    loadingDiv.textContent = "Computing similarity…";
    data.forEach(r => {
        r.score = cosine(userVec, r.embedding);
    });

    const top = data.sort((a, b) => b.score - a.score).slice(0, 3);

    document.getElementById("results").innerHTML = top.map(r => `
        <div class="recipe-card">
            <h3>${r.cuisine.toUpperCase()}</h3>
            <b>Score:</b> ${r.score.toFixed(4)}<br>
            <b>Ingredients:</b> ${r.ingredients.join(", ")}
        </div>
    `).join("");

    loadingDiv.textContent = "";
}

document.getElementById("searchBtn").onclick = recommend;
