// frontend/src/App.jsx
import { useEffect, useState } from "react";
import {
  Brain,
  Settings,
  FileText,
  BarChart3,
  Upload,
  Search,
  AlertTriangle,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import "./App.css";

const API_URL = "http://localhost:8000";

function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [activeTab, setActiveTab] = useState("individual");

  const [manualMode, setManualMode] = useState(false);
  const [anorexiaThreshold, setAnorexiaThreshold] = useState(0.7);
  const [controlThreshold, setControlThreshold] = useState(0.35);
  const [minWords, setMinWords] = useState(3);

  const [text, setText] = useState("");
  const [result, setResult] = useState(null);

  const [compareText, setCompareText] = useState("");
  const [compareResults, setCompareResults] = useState([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [selectedFile, setSelectedFile] = useState(null);
  const [fileTextColumn, setFileTextColumn] = useState("");
  const [fileResultUrl, setFileResultUrl] = useState("");
  const [fileResultName, setFileResultName] = useState("");
  const [fileResults, setFileResults] = useState(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_URL}/models`);
      const data = await res.json();

      setModels(data);

      if (data.length > 0) {
        setSelectedModel(data[0].key);
      }
    } catch (err) {
      setError("No se pudieron cargar los modelos.");
    }
  };

  const selectedModelInfo = models.find((m) => m.key === selectedModel);

  const getConfig = () => {
    if (!manualMode) {
      return {
        anorexia_threshold: 0.7,
        control_threshold: 0.35,
        min_words: 3,
      };
    }

    return {
      anorexia_threshold: Number(anorexiaThreshold),
      control_threshold: Number(controlThreshold),
      min_words: Number(minWords),
    };
  };

  const classifyText = async () => {
    setError("");
    setResult(null);

    if (!text.trim()) {
      setError("Por favor escribe un texto antes de clasificar.");
      return;
    }

    if (!selectedModel) {
      setError("Selecciona un modelo antes de clasificar.");
      return;
    }

    try {
      setLoading(true);

      const res = await fetch(`${API_URL}/predict-text`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text,
          model_key: selectedModel,
          ...getConfig(),
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(JSON.stringify(data));
        return;
      }

      setResult(data);
    } catch (err) {
      setError("Error al conectar con el backend.");
    } finally {
      setLoading(false);
    }
  };

  const compareModels = async () => {
    setError("");
    setCompareResults([]);

    if (!compareText.trim()) {
      setError("Escribe un texto para comparar.");
      return;
    }

    try {
      setLoading(true);

      const responses = await Promise.all(
        models.map(async (model) => {
          const res = await fetch(`${API_URL}/predict-text`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              text: compareText,
              model_key: model.key,
              ...getConfig(),
            }),
          });

          const data = await res.json();

          return {
            ...data,
            short_label: model.short_label,
            family: model.family,
          };
        })
      );

      setCompareResults(responses);
    } catch (err) {
      setError("Error al comparar los modelos.");
    } finally {
      setLoading(false);
    }
  };

  const predictionClass = (label) => {
    if (label === "anorexia") return "pill danger";
    if (label === "control") return "pill success";
    return "pill warning";
  };

  const classifyFile = async () => {
    setError("");
    setFileResultUrl("");
    setFileResultName("");

    if (!selectedFile) {
      setError("Selecciona un archivo CSV o Excel.");
      return;
    }

    if (!selectedModel) {
      setError("Selecciona un modelo antes de clasificar.");
      return;
    }

    try {
      setLoading(true);

      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("model_key", selectedModel);

      if (fileTextColumn.trim()) {
        formData.append("text_column", fileTextColumn.trim());
      }

      const config = getConfig();
      formData.append("anorexia_threshold", config.anorexia_threshold);
      formData.append("control_threshold", config.control_threshold);
      formData.append("min_words", config.min_words);

      const res = await fetch(`${API_URL}/predict-file`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const data = await res.json();
        setError(JSON.stringify(data));
        return;
      }

      const data = await res.json();

      if (data.error) {
        setError(data.error);
        return;
      }

      setFileResults(data);

      const blob = new Blob([data.csv], { type: "text/csv;charset=utf-8;" });
      const url = window.URL.createObjectURL(blob);

      setFileResultUrl(url);
      setFileResultName(data.filename);
    } catch (err) {
      setError("Error al clasificar el archivo.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app">
      <aside className="sidebar">
        <div className="sidebar-title">
          <Settings size={22} />
          <h2>Configuración</h2>
        </div>

        <label>Modelo activo</label>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
        >
          {models.map((model) => (
            <option key={model.key} value={model.key}>
              {model.label}
            </option>
          ))}
        </select>

        {selectedModelInfo && (
          <div className="model-info">
            <p>
              <b>Tipo:</b> {selectedModelInfo.type}
            </p>
            <p>
              <b>Familia:</b> {selectedModelInfo.family}
            </p>
            <p>{selectedModelInfo.description}</p>
          </div>
        )}

        <div className="toggle-row">
          <span>Usar configuración manual</span>
          <input
            type="checkbox"
            checked={manualMode}
            onChange={(e) => setManualMode(e.target.checked)}
          />
        </div>

        {!manualMode ? (
          <div className="info-box">
            <p>Modo automático activo</p>
            <p>Umbral anorexia = 0.70</p>
            <p>Umbral control = 0.35</p>
            <p>Mínimo de palabras = 3</p>
          </div>
        ) : (
          <div className="manual-config">
            <label>Umbral anorexia: {anorexiaThreshold}</label>
            <input
              type="range"
              min="0.5"
              max="0.95"
              step="0.05"
              value={anorexiaThreshold}
              onChange={(e) => setAnorexiaThreshold(e.target.value)}
            />

            <label>Umbral control: {controlThreshold}</label>
            <input
              type="range"
              min="0.05"
              max="0.5"
              step="0.05"
              value={controlThreshold}
              onChange={(e) => setControlThreshold(e.target.value)}
            />

            <label>Mínimo de palabras: {minWords}</label>
            <input
              type="range"
              min="1"
              max="10"
              step="1"
              value={minWords}
              onChange={(e) => setMinWords(e.target.value)}
            />
          </div>
        )}
      </aside>

      <section className="content">
        <header className="hero">
          <div className="hero-icon">
            <Brain size={36} />
          </div>

          <div>
            <h1>Detector de desórdenes alimenticios</h1>
            <p>
              Esta herramienta permite clasificar textos individuales o archivos
              completos para estimar si se detectan señales asociadas a anorexia.
            </p>
          </div>
        </header>

        {selectedModelInfo && (
          <section className="hero-box">
            <b>Modelo cargado:</b> {selectedModelInfo.label}
            <p>
              Puedes clasificar textos individuales, archivos y comparar el mismo
              texto entre todos los modelos disponibles.
            </p>
          </section>
        )}

        <div className="chips">
          {models.map((model) => (
            <span key={model.key}>
              {model.short_label} · {model.family}
            </span>
          ))}
        </div>

        <nav className="tabs">
          <button
            className={activeTab === "individual" ? "active" : ""}
            onClick={() => setActiveTab("individual")}
          >
            <Search size={18} />
            Clasificación individual
          </button>

          <button
            className={activeTab === "archivo" ? "active" : ""}
            onClick={() => setActiveTab("archivo")}
          >
            <Upload size={18} />
            Clasificación por archivo
          </button>

          <button
            className={activeTab === "comparacion" ? "active" : ""}
            onClick={() => setActiveTab("comparacion")}
          >
            <BarChart3 size={18} />
            Comparación entre modelos
          </button>
        </nav>

        {error && (
          <div className="alert">
            <AlertTriangle size={20} />
            {error}
          </div>
        )}

        {activeTab === "individual" && (
          <section className="card">
            <h2>Clasificación individual</h2>
            <p className="caption">
              Evalúa un texto con el modelo actualmente seleccionado.
            </p>

            <label>Escribe el texto a evaluar</label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Ejemplo: hoy no quiero comer nada porque me siento gorda..."
            />

            <button className="primary-button" onClick={classifyText}>
              {loading ? "Clasificando..." : "Clasificar texto"}
            </button>

            {result && <ResultCard result={result} predictionClass={predictionClass} />}
          </section>
        )}

        {activeTab === "archivo" && (
          <section className="card">
            <h2>Clasificación por archivo</h2>
            <p className="caption">
              Sube un archivo CSV o Excel y clasifícalo usando el modelo activo.
            </p>

            <label>Archivo CSV o Excel</label>
            <input
              type="file"
              accept=".csv,.xlsx"
              onChange={(e) => {
                setSelectedFile(e.target.files[0]);
                setFileResultUrl("");
              }}
            />

            <label>Columna de texto</label>
            <input
              type="text"
              value={fileTextColumn}
              onChange={(e) => setFileTextColumn(e.target.value)}
              placeholder="Ejemplo: text, texto, tweet_text, comentario..."
            />

            <p className="caption">
              Si lo dejas vacío, el backend intentará detectar la columna automáticamente.
            </p>

            <button className="primary-button" onClick={classifyFile}>
              {loading ? "Clasificando archivo..." : "Clasificar archivo"}
            </button>

            {fileResults && (
              <>
                <h3>Resumen general</h3>

                <div className="metrics-grid">
                  <Metric label="Filas totales" value={fileResults.total_rows} />
                  <Metric label="Filas válidas" value={fileResults.valid_rows} />
                  <Metric label="Filas descartadas" value={fileResults.dropped_rows} />
                  <Metric label="Columna texto" value={fileResults.text_column} />
                </div>

                <table>
                  <thead>
                    <tr>
                      <th>Clase predicha</th>
                      <th>Cantidad</th>
                      <th>Porcentaje</th>
                    </tr>
                  </thead>
                  <tbody>
                    {fileResults.summary.map((row, index) => (
                      <tr key={index}>
                        <td>{row.clase_predicha}</td>
                        <td>{row.cantidad}</td>
                        <td>{row.porcentaje}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                <h3>Gráfica de resultados</h3>

                <div className="chart-card">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={fileResults.summary}>
                      <XAxis dataKey="clase_predicha" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="cantidad" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <h3>Resultados detallados</h3>

                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        {Object.keys(fileResults.results[0] || {}).map((col) => (
                          <th key={col}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {fileResults.results.slice(0, 30).map((row, index) => (
                        <tr key={index}>
                          {Object.values(row).map((value, i) => (
                            <td key={i}>{String(value)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <p className="caption">
                  Mostrando las primeras 30 filas. Puedes descargar el CSV completo.
                </p>
              </>
            )}

            {fileResultUrl && (
              <div className="download-box">
                <p>Archivo clasificado correctamente.</p>

                <a href={fileResultUrl} download={fileResultName}>
                  Descargar resultados CSV
                </a>
              </div>
            )}
          </section>
        )}

        {activeTab === "comparacion" && (
          <section className="card">
            <h2>Comparación entre modelos</h2>
            <p className="caption">
              Evalúa el mismo texto con todos los modelos disponibles.
            </p>

            <textarea
              value={compareText}
              onChange={(e) => setCompareText(e.target.value)}
              placeholder="Ejemplo: llevo dos días ayunando porque me siento enorme..."
            />

            <button className="primary-button" onClick={compareModels}>
              {loading ? "Comparando..." : "Comparar modelos"}
            </button>

            {compareResults.length > 0 && (
              <>
                <div className="compare-grid">
                  {compareResults.map((item, index) => (
                    <ResultCard
                      key={index}
                      result={item}
                      compact
                      predictionClass={predictionClass}
                    />
                  ))}
                </div>

                <h3>Resumen comparativo</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Modelo</th>
                      <th>Predicción</th>
                      <th>Prob. anorexia</th>
                      <th>Confianza</th>
                      <th>Palabras</th>
                    </tr>
                  </thead>
                  <tbody>
                    {compareResults.map((item, index) => (
                      <tr key={index}>
                        <td>{item.model_label}</td>
                        <td>{item.predicted_label}</td>
                        <td>{formatProbability(item.probability_anorexia)}</td>
                        <td>{item.confidence}</td>
                        <td>{item.word_count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <h3>Gráfica comparativa</h3>
                <div className="chart-card">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={compareResults.map((item) => ({
                        modelo: item.short_label || item.model_label,
                        probabilidad:
                          item.probability_anorexia === null ||
                          item.probability_anorexia === undefined
                            ? 0
                            : Number(item.probability_anorexia),
                      }))}
                    >
                      <XAxis dataKey="modelo" />
                      <YAxis domain={[0, 1]} />
                      <Tooltip />
                      <Bar dataKey="probabilidad" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}
          </section>
        )}
      </section>
    </main>
  );
}

function ResultCard({ result, compact = false, predictionClass }) {
  return (
    <section className={compact ? "result-card compact" : "result-card"}>
      <span className={predictionClass(result.predicted_label)}>
        Predicción: {result.predicted_label}
      </span>

      <div className="metrics-grid">
        <Metric label="Confianza" value={result.confidence} />
        <Metric
          label="Prob. anorexia"
          value={formatProbability(result.probability_anorexia)}
        />
        <Metric label="Palabras" value={result.word_count} />
        <Metric
          label="Cobertura"
          value={
            result.vocab_coverage === null || result.vocab_coverage === undefined
              ? "N/A"
              : Number(result.vocab_coverage).toFixed(2)
          }
        />
      </div>

      <p>
        <b>Mensaje:</b> {result.message}
      </p>

      <p>
        <b>Observaciones:</b> {result.observations}
      </p>

      {!compact && (
        <>
          <h3>Texto procesado</h3>
          <pre>{result.cleaned_text}</pre>
        </>
      )}
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function formatProbability(value) {
  if (value === null || value === undefined) return "N/A";
  return Number(value).toFixed(4);
}

export default App;