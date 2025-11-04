document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("verify-form");
  const fileInput = document.getElementById("imageInput");
  const fileName = document.getElementById("file-name");
  const preview = document.getElementById("preview");
  const resultDiv = document.getElementById("result");
  const status = document.getElementById("status");
  const details = document.getElementById("details");

  // Mostrar vista previa y nombre del archivo
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;

    fileName.textContent = `📁 ${file.name}`;
    fileName.classList.remove("hidden");

    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      preview.classList.remove("hidden");
    };
    reader.readAsDataURL(file);
  });

  // Envío del formulario
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const file = fileInput.files[0];
    if (!file) return alert("Selecciona una imagen primero.");

    const formData = new FormData();
    formData.append("file", file);

    resultDiv.classList.remove("hidden");
    status.textContent = "Procesando...";
    status.className = "text-blue-400 text-xl font-bold animate-pulse";
    details.textContent = "";

    try {
      const res = await fetch("/verify", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Error en la verificación.");

      if (data.is_me === true) {
        status.textContent = "🟢 Yo";
        status.className = "text-green-400 text-4xl font-extrabold";
      } else {
        status.textContent = "🔴 No yo";
        status.className = "text-red-400 text-4xl font-extrabold";
      }

      details.textContent = `Score: ${data.score.toFixed(3)} | Umbral: ${data.threshold} | Tiempo: ${data.timing_ms.toFixed(1)} ms`;
    } catch (err) {
      console.error(err);
      status.textContent = "❌ Error procesando la imagen";
      status.className = "text-red-400 text-xl font-bold";
      details.textContent = "Revisa la consola para más detalles.";
    }
  });
});
