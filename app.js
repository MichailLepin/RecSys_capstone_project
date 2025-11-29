let embedder;
let modelReady = false;

const loadingDiv = document.getElementById("loading");

// --------------------------
// Load model (Xenova MiniLM)
// --------------------------
async function loadModel() {
  loadingDiv.innerText = "Loading model… This may take ~10–20 seconds";

  embedder = await window.pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );

  modelReady = true;
  loadingDiv.innerText = "Model loaded ✔";
}

loadModel();


// --------------------------
// Cosine similarity
// --------------------------
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}


// --------------------------
// Recommend recipes
// --------------------------
async function recommend() {
  if (!modelReady) {
    alert("Model is still loading. Please wait...");
    return;
  }

  const userText = document.getElementById("ingredientsInput").value.trim();
  if (!userText) return;

  loadingDiv.innerText = "Embedding your ingredients…";

  // 1. Embed the user input
  const output = await embedder(userText);
  const userEmbedding = Array.from(output.data[0]);

  // 2. Load recipes (from GitHub Release)
  loadingDiv.innerText = "Fetching recipes… This may take a bit";

  const recipesURL =
    "https://github.com/miketernov/RecSys_capstone_project/releases/download/v1/recipes_with_embeddings.json";

  const recipes = await fetch(recipesURL).then(r => r.json());

  // 3. Compute similarity
  loadingDiv.innerText = "Computing similarity…";

  recipes.forEach(r => {
    r.score = cosine(userEmbedding, r.embedding);
  });

  // 4. Pick top 3
  const top = recipes.sort((a, b) => b.score - a.score).slice(0, 3);

  // 5. Render
  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = top
    .map(
      r => `
    <div class="recipe-card">
      <h3>${r.cuisine.toUpperCase()}</h3>
      <b>Score:</b> ${r.score.toFixed(4)}<br/>
      <div class="ingredients"><b>Ingredients:</b> ${r.ingredients.join(", ")}</div>
    </div>
  `
    )
    .join("");

  loadingDiv.innerText = "";
}

document.getElementById("searchBtn").onclick = recommend;
