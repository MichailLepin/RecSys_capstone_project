// =====================================
// CONFIG
// =====================================
const TOTAL_CHUNKS = 17;              // сколько кусков recipes_with_embeddings
let recipes = [];                     // сюда складываем рецепты
let recipeEmbeddings = [];            // сюда – эмбеддинги (Float32Array)
let embedder = null;                  // модель для эмбеддингов запроса

const loading = document.getElementById("loading");
const progress = document.getElementById("progress");
const resultsDiv = document.getElementById("results");
const searchBtn = document.getElementById("searchBtn");


// =====================================
// 1) Загрузка рецептов с эмбеддингами
// (chunks/part1.json, part2.json, ...)
// Каждый рецепт: { id, cuisine, ingredients, embedding: [..] }
// =====================================
async function loadChunks() {
    if (recipes.length > 0) return;

    let all = [];
    for (let i = 1; i <= TOTAL_CHUNKS; i++) {
        progress.textContent = `Loading recipes ${i}/${TOTAL_CHUNKS}…`;
        const chunk = await fetch(`chunks/part${i}.json`).then(r => r.json());

        chunk.forEach(r => {
            all.push({
                id: r.id,
                cuisine: r.cuisine,
                ingredients: r.ingredients
            });
            recipeEmbeddings.push(new Float32Array(r.embedding));
        });
    }

    recipes = all;
    progress.textContent = "";
    console.log("Loaded recipes:", recipes.length);
}


// =====================================
// 2) Загрузка модели эмбеддингов в браузере
// Используем @xenova/transformers
// Модель: all-MiniLM-L6-v2
// =====================================
async function loadModel() {
    if (embedder) return;

    loading.textContent = "Loading embedding model… (first time is slow)";
    const { pipeline } = window.transformers;

    // feature-extraction с mean-пулингом и нормализацией
    embedder = await pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2"
    );

    loading.textContent = "Model loaded ✔";
    console.log("Model ready");
}


// =====================================
// 3) Косинусное сходство
// =====================================
function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
        const va = a[i];
        const vb = b[i];
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
    }
    if (na === 0 || nb === 0) return 0;
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}


// =====================================
// 4) Получить эмбеддинг текста пользователя
// Важно написать строку так же, как в Python:
// "Ingredients: tomato, cheese, lettuce"
// =====================================
async function embedUserIngredients(text) {
    loading.textContent = "Embedding user ingredients…";

    const fullText = "Ingredients: " + text.toLowerCase();

    // pooling и нормализация есть прямо в options (transformers.js)
    const output = await embedder(fullText, {
        pooling: "mean",
        normalize: true
    });

    // output.data — Float32Array
    return output.data;
}


// =====================================
// 5) Сгенерировать простое объяснение
// =====================================
function makeExplanation(recipe, score, userText) {
    const userTokens = userText
        .toLowerCase()
        .split(/[\s,()\/-]+/)
        .filter(t => t.length > 1);

    const recipeTokens = new Set(
        recipe.ingredients
            .join(" ")
            .toLowerCase()
            .split(/[\s,()\/-]+/)
            .filter(t => t.length > 1)
    );

    const overlap = [...new Set(userTokens.filter(t => recipeTokens.has(t)))];

    let s = `Cosine similarity: ${(score * 100).toFixed(1)}%. `;
    if (overlap.length > 0) {
        s += `Совпадающие ингредиенты: ${overlap.join(", ")}. `;
    }
    s += `Кухня: ${recipe.cuisine.toUpperCase()}.`;

    return s;
}


// =====================================
// 6) Основная функция рекомендации
// =====================================
async function recommend() {
    const txt = document.getElementById("ingredientsInput").value.trim();
    if (!txt) return;

    resultsDiv.innerHTML = "";
    loading.textContent = "Preparing recommendation…";

    // На всякий случай – дождаться загрузки модели и данных
    await loadChunks();
    await loadModel();

    // Эмбеддинг запроса
    const userVec = await embedUserIngredients(txt);

    // Считаем косинус с каждым рецептом
    const scored = recipeEmbeddings.map((emb, idx) => ({
        recipe: recipes[idx],
        score: cosine(userVec, emb)
    }));

    // Топ-3 по схожести
    const top = scored.sort((a, b) => b.score - a.score).slice(0, 3);

    // Рендер карточек
    resultsDiv.innerHTML = top.map(x => `
        <div class="recipe-card">
            <h3>${x.recipe.cuisine.toUpperCase()}</h3>
            <p><b>Score:</b> ${x.score.toFixed(4)}</p>
            <p><b>Ingredients:</b> ${x.recipe.ingredients.join(", ")}</p>
            <p><i>${makeExplanation(x.recipe, x.score, txt)}</i></p>
        </div>
    `).join("");

    loading.textContent = "";
}


// =====================================
// 7) Инициализация при загрузке
// (загружаем рецепты, потом модель)
// =====================================
async function init() {
    try {
        loading.textContent = "Loading recipes…";
        await loadChunks();

        // Модель можно грузить лениво (при первом поиске),
        // но для демки лучше сразу:
        await loadModel();

        loading.textContent = "Ready ✔ Type ingredients and click Find Recipes";
    } catch (e) {
        console.error(e);
        loading.textContent = "Error during init (see console)";
    }
}

init();
searchBtn.onclick = recommend;
