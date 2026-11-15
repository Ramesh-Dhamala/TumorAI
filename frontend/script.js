const API_URL = "https://tumorai-api.onrender.com";

const form = document.getElementById("prediction-form");
const featuresContainer = document.getElementById("features-container");
const predictButton = document.getElementById("predict-button");
const demoButton = document.getElementById("demo-button");

const filledCount = document.getElementById("filled-count");
const formProgress = document.getElementById("form-progress");

const resultSection = document.getElementById("result-section");
const resultCard = document.getElementById("result-card");
const predictionResult = document.getElementById("prediction-result");
const resultDescription = document.getElementById("result-description");
const resultIcon = document.getElementById("result-icon");

const probabilityValue = document.getElementById("probability-value");
const probabilityBar = document.getElementById("probability-bar");

const errorMessage = document.getElementById("error-message");

// --------------------------------------------------
// 1. Format feature labels
// --------------------------------------------------

function formatLabel(name) {
  return name
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

// --------------------------------------------------
// 2. Group features into Mean, SE and Worst
// --------------------------------------------------

function groupFeatures(features) {
  return {
    mean: features.filter((feature) => feature.endsWith("_mean")),
    se: features.filter((feature) => feature.endsWith("_se")),
    worst: features.filter((feature) => feature.endsWith("_worst")),
  };
}

// --------------------------------------------------
// 3. Create input fields dynamically
// --------------------------------------------------

function createFeatureField(feature) {
  const field = document.createElement("div");
  field.className = "feature-field";

  const label = document.createElement("label");
  label.htmlFor = feature;
  label.textContent = formatLabel(feature);

  const input = document.createElement("input");

  input.type = "number";
  input.id = feature;
  input.name = feature;
  input.step = "any";
  input.required = true;
  input.placeholder = "Enter value";

  input.addEventListener("input", updateProgress);

  field.appendChild(label);
  field.appendChild(input);

  return field;
}

// --------------------------------------------------
// 4. Render feature groups
// --------------------------------------------------

function renderFeatures(features) {
  const groups = groupFeatures(features);

  const groupInformation = [
    {
      key: "mean",
      number: "01",
      title: "Mean measurements",
      description: "Average values",
    },
    {
      key: "se",
      number: "02",
      title: "Standard error measurements",
      description: "Measurement variation",
    },
    {
      key: "worst",
      number: "03",
      title: "Worst measurements",
      description: "Largest values",
    },
  ];

  featuresContainer.innerHTML = "";

  groupInformation.forEach((group) => {
    const groupElement = document.createElement("section");
    groupElement.className = "feature-group";

    const heading = document.createElement("div");
    heading.className = "group-heading";

    heading.innerHTML = `
            <span class="group-number">${group.number}</span>
            <h3>${group.title}</h3>
            <span>${group.description}</span>
            <div class="group-line"></div>
        `;

    const grid = document.createElement("div");
    grid.className = "features-grid";

    groups[group.key].forEach((feature) => {
      grid.appendChild(createFeatureField(feature));
    });

    groupElement.appendChild(heading);
    groupElement.appendChild(grid);

    featuresContainer.appendChild(groupElement);
  });

  updateProgress();
}

// --------------------------------------------------
// 5. Update form progress
// --------------------------------------------------

function updateProgress() {
  const inputs = [...form.querySelectorAll("input[type='number']")];

  const completed = inputs.filter((input) => {
    return input.value.trim() !== "" && Number.isFinite(input.valueAsNumber);
  }).length;

  const percentage = (completed / inputs.length) * 100;

  filledCount.textContent = completed;
  formProgress.style.width = `${percentage}%`;
}

// --------------------------------------------------
// 6. Load model features from FastAPI
// --------------------------------------------------

async function loadFeatures() {
  try {
    const response = await fetch(`${API_URL}/features`);

    if (!response.ok) {
      throw new Error("Unable to retrieve model features.");
    }

    const data = await response.json();

    renderFeatures(data.features);

    predictButton.disabled = false;
  } catch (error) {
    featuresContainer.innerHTML = "";

    errorMessage.textContent =
      "Backend connection failed. Make sure FastAPI is running.";

    errorMessage.classList.remove("hidden");
  }
}

// --------------------------------------------------
// 7. Display prediction result
// --------------------------------------------------

function displayResult(data) {
  const isMalignant = data.prediction === 1;

  resultCard.classList.toggle("malignant", isMalignant);

  predictionResult.textContent = data.result;

  resultIcon.textContent = isMalignant ? "!" : "✓";

  resultDescription.textContent = isMalignant
    ? "The model classified this input as malignant."
    : "The model classified this input as benign.";

  const probability = Math.min(
    100,
    Math.max(0, Number(data.malignant_probability)),
  );

  probabilityValue.textContent = `${probability}%`;

  probabilityBar.style.width = "0%";

  resultSection.classList.remove("hidden");

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      probabilityBar.style.width = `${probability}%`;
    });
  });

  resultSection.scrollIntoView({
    behavior: "smooth",
    block: "center",
  });
}

// --------------------------------------------------
// 8. Submit prediction request
// --------------------------------------------------

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  errorMessage.classList.add("hidden");
  resultSection.classList.add("hidden");

  predictButton.disabled = true;

  predictButton.querySelector(".button-text").textContent =
    "Analyzing measurements...";

  try {
    const formData = new FormData(form);
    const features = {};

    for (const [key, value] of formData.entries()) {
      if (value.trim() === "") {
        throw new Error(`Please enter ${formatLabel(key)}.`);
      }

      const numberValue = Number(value);

      if (!Number.isFinite(numberValue)) {
        throw new Error(`Invalid value for ${formatLabel(key)}.`);
      }

      features[key] = numberValue;
    }

    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        features: features,
      }),
    });

    console.log("Total features:", Object.keys(features).length);
    console.log("Feature names:", Object.keys(features));

    const data = await response.json();

    console.log("Prediction response:", data);

    if (!response.ok) {
      console.error("Backend validation error:", data.detail);
      throw new Error(JSON.stringify(data.detail));
    }

    displayResult(data);
  } catch (error) {
    errorMessage.textContent = error.message;
    errorMessage.classList.remove("hidden");
  } finally {
    predictButton.disabled = false;

    predictButton.querySelector(".button-text").textContent =
      "Generate prediction";
  }
});

// --------------------------------------------------
// 9. Reset form
// --------------------------------------------------

form.addEventListener("reset", () => {
  resultSection.classList.add("hidden");
  errorMessage.classList.add("hidden");

  setTimeout(updateProgress, 0);
});

// --------------------------------------------------
// 10. Initialize application
// --------------------------------------------------

loadFeatures();

// --------------------------------------------------
// Demo: Load a real sample from the dataset
// --------------------------------------------------

demoButton.addEventListener("click", async () => {
  errorMessage.classList.add("hidden");
  resultSection.classList.add("hidden");

  demoButton.disabled = true;
  demoButton.textContent = "Loading sample...";

  try {
    const response = await fetch(`${API_URL}/demo`);

    if (!response.ok) {
      throw new Error("Unable to load demo measurements.");
    }

    const data = await response.json();

    // Fill all 30 input fields
    Object.entries(data.features).forEach(([key, value]) => {
      const input = document.getElementById(key);

      if (input) {
        input.value = value;
      }
    });

    // Update completed input counter
    updateProgress();
  } catch (error) {
    errorMessage.textContent = error.message;
    errorMessage.classList.remove("hidden");
  } finally {
    demoButton.disabled = false;
    demoButton.textContent = "✨ Try Demo";
  }
});
