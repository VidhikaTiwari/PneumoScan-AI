import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

function Home() {
  const navigate = useNavigate();

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) return;

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setError("");
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError("Please upload an X-ray image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      setLoading(true);
      setError("");

      const response = await axios.post("http://127.0.0.1:8000/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      navigate("/results", {
        state: {
          result: response.data,
          previewUrl: previewUrl,
        },
      });
    } catch (err) {
      console.error(err);
      setError("Something went wrong while analyzing the X-ray.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="home-page">
      <div className="background-overlay"></div>

      <div className="hero-card">
        <h1>PneumoScan AI</h1>
        <p className="tagline">
          AI-powered chest X-ray screening for pneumonia detection
        </p>

        <label className="upload-label">
          Upload X-ray
          <input type="file" accept="image/*" onChange={handleFileChange} hidden />
        </label>

        {previewUrl && (
          <div className="preview-container">
            <img src={previewUrl} alt="X-ray preview" className="preview-image" />
          </div>
        )}

        {selectedFile && (
            <button
                onClick={handleAnalyze}
                disabled={loading}
                className="analyze-btn"
            >
                {loading ? "Analyzing..." : "Analyze Scan"}
            </button>
            )}

                {error && <p className="error-text">{error}</p>}
            </div>
            </div>
        );
        }

export default Home;