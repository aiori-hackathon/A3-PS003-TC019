// Traffix frontend
// Keep the FastAPI backend running at this address.
const API_BASE = "http://127.0.0.1:8000";

const navItems = document.querySelectorAll(".nav-item");
const pages = document.querySelectorAll(".page");
const pageName = document.getElementById("page-name");

function showPage(pageId) {
  pages.forEach(page => page.classList.toggle("active", page.id === pageId));
  navItems.forEach(item => item.classList.toggle("active", item.dataset.page === pageId));
  const activeItem = document.querySelector(`[data-page="${pageId}"]`);
  if (pageName) pageName.textContent = activeItem ? activeItem.textContent.trim() : "Overview";
  window.scrollTo({ top: 0, behavior: "smooth" });
}
window.showPage = showPage;
navItems.forEach(item => item.addEventListener("click", () => showPage(item.dataset.page)));

const today = document.getElementById("today");
if (today) today.textContent = new Date().toLocaleDateString("en-IN", {day:"numeric", month:"long", year:"numeric"});

const uploadInput = document.getElementById("image-upload");
const previewWrap = document.getElementById("preview-wrap");
const previewImage = document.getElementById("image-preview");
const fileName = document.getElementById("file-name");
const runButton = document.getElementById("run-detection");
const statusBox = document.getElementById("detection-message");
const previewStatus = document.getElementById("preview-status");
const resultsPanel = document.getElementById("detection-results");
const classCounts = document.getElementById("class-counts");
const detectionList = document.getElementById("detection-list");
const detectionTime = document.getElementById("detection-time");
let selectedFile = null;
let imageUrl = null;

async function checkBackend() {
  if (!statusBox) return;
  try {
    const response = await fetch(`${API_BASE}/api/health`);
    if (!response.ok) throw new Error("Backend health check failed");
    const data = await response.json();
    statusBox.textContent = data.model_loaded
      ? "Backend connected · Federated global model loaded and ready."
      : "Backend connected, but the model is not loaded. Check backend/models/federated_global.pt.";
    statusBox.classList.toggle("notice-success", !!data.model_loaded);
  } catch (error) {
    statusBox.textContent = "Backend not connected. Start the FastAPI server, then refresh this page.";
  }
}

if (uploadInput) {
  uploadInput.addEventListener("change", event => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!["image/jpeg", "image/png"].includes(file.type)) {
      alert("Please select a JPG or PNG image.");
      uploadInput.value = "";
      return;
    }
    selectedFile = file;
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    imageUrl = URL.createObjectURL(file);
    previewImage.src = imageUrl;
    fileName.textContent = file.name;
    previewStatus.textContent = "Image selected · ready for inference";
    previewWrap.classList.remove("hidden");
    runButton.disabled = false;
    resultsPanel.classList.add("hidden");
    detectionTime.textContent = "";
  });
}

if (runButton) {
  runButton.addEventListener("click", async () => {
    if (!selectedFile) return;
    runButton.disabled = true;
    runButton.textContent = "Running inference…";
    statusBox.textContent = "Sending image to the YOLO26n federated model…";
    previewStatus.textContent = "Running model inference";
    const formData = new FormData();
    formData.append("file", selectedFile);
    try {
      const start = performance.now();
      const response = await fetch(`${API_BASE}/api/detect`, { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Detection request failed");
      const elapsed = ((performance.now() - start) / 1000).toFixed(2);
      const resultImage = payload.image || payload.annotated_image;
      if (resultImage) {
        previewImage.src = resultImage.startsWith("data:")
          ? resultImage
          : `data:image/jpeg;base64,${resultImage}`;
      }
      const counts = payload.class_counts || payload.counts || {};
      classCounts.innerHTML = "";
      const entries = Object.entries(counts);
      if (entries.length) {
        entries.forEach(([name, count]) => {
          const item = document.createElement("div");
          item.className = "count-card";
          const label = document.createElement("span");
          label.textContent = name;
          const value = document.createElement("strong");
          value.textContent = count;
          item.append(label, value);
          classCounts.appendChild(item);
        });
      } else {
        classCounts.textContent = "No vehicles detected in this image.";
      }
      detectionList.innerHTML = "";
      const detections = payload.detections || payload.boxes || [];
      detections.slice(0, 30).forEach((detection, index) => {
        const row = document.createElement("div");
        row.className = "detection-row";
        const name = document.createElement("span");
        name.textContent = detection.class_name || detection.name || detection.label || `Detection ${index + 1}`;
        const confidence = document.createElement("strong");
        const conf = detection.confidence ?? detection.conf;
        confidence.textContent = conf == null ? "" : `${(Number(conf) * (Number(conf) <= 1 ? 100 : 1)).toFixed(1)}% confidence`;
        row.append(name, confidence);
        detectionList.appendChild(row);
      });
      resultsPanel.classList.remove("hidden");
      detectionTime.textContent = `Completed in ${payload.inference_time ?? elapsed} seconds`;
      previewStatus.textContent = "Detection complete · annotated output";
      statusBox.textContent = "Inference completed successfully.";
      statusBox.classList.add("notice-success");
    } catch (error) {
      statusBox.textContent = `Detection failed: ${error.message}`;
      previewStatus.textContent = "Inference failed";
    } finally {
      runButton.disabled = false;
      runButton.textContent = "Run vehicle detection ↗";
    }
  });
}
checkBackend();
