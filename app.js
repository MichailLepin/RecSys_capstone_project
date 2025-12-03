// ==========================
// Глобальные переменные
// ==========================
const TOTAL_CHUNKS = 17;      // сколько частей ты сделал в Python
let recipes = [];             // сюда загрузим все рецепты (id, cuisine, ingredients, embedding)
let embedder = null;          // LLM-модель в браузере
let modelReady = false;

const loading = document.getElementById("loading");
const progress = document.getElementById("progress");
const resultsDiv = document.getElementById("results");

// ==========================
// 1) Загрузка чанков с GitHub
// ==========================
async function loadChunks() {
    if (recipes.length > 0) return;

    let all = [];
    for (let i = 1; i <= TOTAL_CHUNKS; i++) {
        progress.textContent = `Loading chunk ${i}/${TOTAL_CHUNKS}…`;
        const url = `chunks/part${i}.json`;
        const chunk = await fetch(url).then(r => r.json());
        all.push(...chunk);
    }

    recipes = all;
    progress.textContent = `Loaded recipes: ${recipes.length}`;
    console.log("Loaded recipes:", recipes.length);
}

// ==========================
// 2) Загрузка модели эмбеддингов
// ==========================
async function loadModel() {
    try {
        const { pipeline } = window.transformers;
        // feature-extraction даёт эмбеддинг текста
        embedder = await pipeline(
            "feature-extraction",
            "Xenova/all-MiniLM-L6-v2"
        );
        modelReady = true;
        console.log("Embedding model loaded");
    } catch (e) {
        console.error("Error loading model:", e);
        loading.textContent = "Error loading embedding model (see console)";
    }
}

// ==========================
// 3) Косинусное сходство
// ==========================
function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    const len = a.length;
    for (let i = 0; i < len; i++) {
        const va = a[i];
        const vb = b[i];
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
    }
    if (na === 0 || nb === 0) return 0;
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ==========================
// 4) Объяснение для пользователя
// ==========================
function buildExplanation(query, recipe) {
    const qTokens = new Set(
        query.toLowerCase().split(/[^a-zа-яё]+/).filter(t => t.length > 1)
    );

    const overlap = new Set();
    recipe.ingredients.forEach(ing => {
        const lowIng = ing.toLowerCase();
        qTokens.forEach(t => {
            if (lowIng.includes(t)) {
                overlap.add(t);
            }
        });
    });

    const shared = Array.from(overlap);
    let reason = `This is a ${recipe.cuisine.toUpperCase()} recipe. `;

    if (shared.length > 0) {
        reason += `It uses your ingredients: ${shared.join(", ")}.`;
    } else {
        reason += `It is semantically close to your ingredient combination.`;
    }

    return reason;
}

// ==========================
// 5) Рекомендации
// ==========================
async function recommend() {
    const txt = document.getElementById("ingredientsInput").value.trim();
    if (!txt) return;

    if (!modelReady) {
        alert("Embedding model is still loading, please wait a bit.");
        return;
    }

    loading.textContent = "Embedding user ingredients… (first time is slow)";
    resultsDiv.innerHTML = "";

    try {
        // эмбеддинг запроса, mean pooling + нормализация внутри модели
        const output = await embedder(txt, {
            pooling: "mean",
            normalize: true
        });
        const userVec = Array.from(output.data); // Float32Array → обычный массив

        // считаем сходство с эмбеддингами рецептов
        const scores = recipes.map(r => ({
            recipe: r,
            score: cosine(userVec, r.embedding)
        }));

        // top-3
        scores.sort((a, b) => b.score - a.score);
        const top = scores.slice(0, 3);

        // рисуем карточки
        resultsDiv.innerHTML = top.map(x => {
            const r = x.recipe;
            const explanation = buildExplanation(txt, r);

            return `
            <div class="recipe-card">
                <h3>${r.cuisine.toUpperCase()}</h3>
                <p><b>Score:</b> ${x.score.toFixed(4)}</p>
                <p><b>Ingredients:</b> ${r.ingredients.join(", ")}</p>
                <p class="explanation">${explanation}</p>
            </div>
            `;
        }).join("");

        loading.textContent = "";
    } catch (e) {
        console.error("Error in recommend:", e);
        loading.textContent = "Error during recommendation (see console)";
    }
}

// ==========================
// 6) Инициализация
// ==========================
async function init() {
    try {
        loading.textContent = "Loading recipes…";
        await loadChunks();

        loading.textContent = "Loading embedding model… (first time is slow)";
        await loadModel();

        if (modelReady) {
            loading.textContent = "Ready ✔ Type your ingredients and press 'Find Recipes'.";
        }
    } catch (e) {
        console.error("Init error:", e);
        loading.textContent = "Error during init (see console)";
    }
}

document.getElementById("searchBtn").onclick = recommend;

// стартуем
init();
