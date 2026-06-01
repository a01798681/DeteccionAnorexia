import { useEffect, useMemo, useState } from "react";
import {
  Brain,
  Settings,
  BarChart3,
  Upload,
  Search,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  Download,
  SlidersHorizontal,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import "./App.css";

const API_URL = "http://localhost:8000";

function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [activeTab, setActiveTab] = useState("individual");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

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

  const [riskTermsText, setRiskTermsText] = useState("");
  const [positiveSafeText, setPositiveSafeText] = useState("");
  const [negationSafeText, setNegationSafeText] = useState("");
  const [termCounts, setTermCounts] = useState(null);
  const [termsMessage, setTermsMessage] = useState("");

  useEffect(() => {
    fetchModels();
    fetchCustomTerms();
  }, []);

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_URL}/models`);
      const data = await res.json();

      const modelsData = data.models || [];
      const defaultModelKey = data.default_model_key || "";

      setModels(modelsData);

      if (defaultModelKey) {
        setSelectedModel(defaultModelKey);
      } else if (modelsData.length > 0) {
        setSelectedModel(modelsData[0].key);
      }
    } catch {
      setError("No se pudieron cargar los modelos.");
    }
  };

  const fetchCustomTerms = async () => {
    try {
      const res = await fetch(`${API_URL}/custom-terms`);
      const data = await res.json();

      const custom = data.custom_terms || {};

      setRiskTermsText((custom.risk_terms_extra || []).join("\n"));
      setPositiveSafeText((custom.positive_safe_terms_extra || []).join("\n"));
      setNegationSafeText((custom.negation_safe_terms_extra || []).join("\n"));
      setTermCounts(data.active_counts || null);
    } catch {
      setError("No se pudieron cargar los términos personalizados.");
    }
  };

  const saveCustomTerms = async () => {
    setError("");
    setTermsMessage("");

    try {
      const res = await fetch(`${API_URL}/custom-terms`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          risk_terms_extra: riskTermsText.split("\n"),
          positive_safe_terms_extra: positiveSafeText.split("\n"),
          negation_safe_terms_extra: negationSafeText.split("\n"),
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError("No se pudieron guardar los términos.");
        return;
      }

      setTermsMessage(data.message || "Términos guardados correctamente.");
      setTermCounts(data.active_counts || null);
    } catch {
      setError("Error al guardar los términos.");
    }
  };

  const selectedModelInfo = useMemo(
    () => models.find((m) => m.key === selectedModel),
    [models, selectedModel]
  );

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
    } catch {
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

      const res = await fetch(`${API_URL}/compare-models`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: compareText,
          ...getConfig(),
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError("Error al comparar los modelos.");
        return;
      }

      setCompareResults(data);
    } catch {
      setError("Error al comparar los modelos.");
    } finally {
      setLoading(false);
    }
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
    } catch {
      setError("Error al clasificar el archivo.");
    } finally {
      setLoading(false);
    }
  };

  const fillExample = (value, tab = "individual") => {
    if (tab === "individual") {
      setActiveTab("individual");
      setText(value);
      return;
    }
    setActiveTab("comparacion");
    setCompareText(value);
  };

  const predictionClass = (label) => {
    if (label === "anorexia") return "badge badge-danger";
    if (label === "control") return "badge badge-success";
    return "badge badge-warning";
  };

  const getModelToneClass = (modelKey = "") => {
    if (modelKey.includes("ensemble")) return "tone-recommended";
    if (modelKey.includes("cascade")) return "tone-advanced";
    if (modelKey.includes("beto")) return "tone-transformer";
    if (modelKey.includes("forest")) return "tone-experimental";
    return "tone-classic";
  };

  return (
    <div className="app-shell">
      <div className="app-layout">
        <aside className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
          <div className="sidebar-top">
            <div className="sidebar-title">
              <Settings size={18} />
              {!sidebarCollapsed && <h2>Configuración</h2>}
            </div>

            <button
              className="icon-button"
              onClick={() => setSidebarCollapsed((prev) => !prev)}
              title={sidebarCollapsed ? "Expandir panel" : "Colapsar panel"}
            >
              {sidebarCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
            </button>
          </div>

          {!sidebarCollapsed && (
            <>
              <div className="sidebar-section">
                <label className="sidebar-label">Modelo activo</label>
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
                  <div className="model-info-card">
                    <p>
                      <b>Tipo:</b> {selectedModelInfo.type}
                    </p>
                    <p>
                      <b>Familia:</b> {selectedModelInfo.family}
                    </p>
                    <p>{selectedModelInfo.description}</p>
                    {selectedModelInfo.recommended && (
                      <div className="recommended-chip">
                        <Sparkles size={14} />
                        <span>Recomendado</span>
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="sidebar-section">
                <div className="sidebar-section-title">
                  <SlidersHorizontal size={16} />
                  <span>Umbrales</span>
                </div>

                <div className="checkbox-row">
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
              </div>

              <div className="sidebar-section">
                <div className="sidebar-section-title">
                  <Brain size={16} />
                  <span>Jerga / palabras nuevas</span>
                </div>

                <label>Términos de riesgo extra</label>
                <textarea
                  value={riskTermsText}
                  onChange={(e) => setRiskTermsText(e.target.value)}
                  placeholder="Uno por línea"
                  rows={6}
                />

                <label>Términos seguros extra</label>
                <textarea
                  value={positiveSafeText}
                  onChange={(e) => setPositiveSafeText(e.target.value)}
                  placeholder="Uno por línea"
                  rows={4}
                />

                <label>Términos de negación segura extra</label>
                <textarea
                  value={negationSafeText}
                  onChange={(e) => setNegationSafeText(e.target.value)}
                  placeholder="Uno por línea"
                  rows={4}
                />

                <div className="two-columns">
                  <button className="btn btn-primary" onClick={saveCustomTerms}>
                    Guardar términos
                  </button>

                  <button className="btn btn-secondary" onClick={fetchCustomTerms}>
                    Recargar términos
                  </button>
                </div>

                {termsMessage && <p className="helper-text success-text">{termsMessage}</p>}

                {termCounts && (
                  <div className="info-box">
                    <p>Términos activos de riesgo: {termCounts.risk_terms}</p>
                    <p>Términos seguros activos: {termCounts.positive_safe_terms}</p>
                    <p>Términos de negación segura: {termCounts.negation_safe_terms}</p>
                  </div>
                )}
              </div>
            </>
          )}
        </aside>

        <main className="main-content">
          <header className="topbar">
            <div className="topbar-left">
              <div className="topbar-icon">
                <Brain size={22} />
              </div>
              <div>
                <h1 className="topbar-title">Detector de desórdenes alimenticios</h1>
                <p className="topbar-subtitle">
                  Clasificación de textos, archivos y comparación entre modelos.
                </p>
              </div>
            </div>
          </header>

          <section className="hero-card">
            <div className="hero-icon">🧠</div>
            <div>
              <h2 className="hero-title">Sistema de análisis de riesgo alimenticio</h2>
              <p className="hero-subtitle">
                Esta herramienta permite clasificar textos individuales o archivos
                completos para estimar si se detectan señales asociadas a anorexia.
              </p>
            </div>
          </section>

          {selectedModelInfo && (
            <section className="status-card">
              <strong>Modelo cargado:</strong> {selectedModelInfo.label}
              <p>
                Puedes clasificar textos individuales, archivos y comparar el mismo
                texto entre todos los modelos disponibles.
              </p>
            </section>
          )}

          <section className="card">
            <div className="model-chip-group">
              {models.map((model) => (
                <button
                  key={model.key}
                  type="button"
                  className={`model-chip ${selectedModel === model.key ? "active" : ""} ${
                    model.recommended ? "recommended" : ""
                  }`}
                  onClick={() => setSelectedModel(model.key)}
                >
                  {model.short_label} · {model.family}
                </button>
              ))}
            </div>
          </section>

          <section className="card">
            <div className="tab-group">
              <button
                className={`tab-chip ${activeTab === "individual" ? "active" : ""}`}
                onClick={() => setActiveTab("individual")}
              >
                <Search size={16} />
                <span>Clasificación individual</span>
              </button>

              <button
                className={`tab-chip ${activeTab === "archivo" ? "active" : ""}`}
                onClick={() => setActiveTab("archivo")}
              >
                <Upload size={16} />
                <span>Clasificación por archivo</span>
              </button>

              <button
                className={`tab-chip ${activeTab === "comparacion" ? "active" : ""}`}
                onClick={() => setActiveTab("comparacion")}
              >
                <BarChart3 size={16} />
                <span>Comparación entre modelos</span>
              </button>
            </div>
          </section>

          {error && (
            <div className="alert-banner">
              <AlertTriangle size={18} />
              <span>{error}</span>
            </div>
          )}

          <section className="card">
            <div className="quick-examples">
              <span className="quick-title">Casos rápidos:</span>

              <button
                className="mini-example"
                onClick={() =>
                  fillExample(
                    "quiero bajar de peso dejando de comer por completo",
                    activeTab === "comparacion" ? "comparacion" : "individual"
                  )
                }
              >
                Riesgo
              </button>

              <button
                className="mini-example"
                onClick={() =>
                  fillExample(
                    "hoy salí a cenar con mis amigos y me sentí tranquila",
                    activeTab === "comparacion" ? "comparacion" : "individual"
                  )
                }
              >
                Neutral
              </button>

              <button
                className="mini-example"
                onClick={() =>
                  fillExample(
                    "bodycheck otra vez, me siento enorme y quiero ayunar",
                    activeTab === "comparacion" ? "comparacion" : "individual"
                  )
                }
              >
                Jerga
              </button>
            </div>
          </section>

          {activeTab === "individual" && (
            <section className="card">
              <h2>Clasificación individual</h2>
              <p className="helper-text">
                Evalúa un texto con el modelo actualmente seleccionado.
              </p>

              <label>Escribe el texto a evaluar</label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Ejemplo: hoy no quiero comer nada porque me siento gorda..."
              />

              <div className="actions-row">
                <button className="btn btn-primary" onClick={classifyText}>
                  {loading ? "Clasificando..." : "Clasificar texto"}
                </button>
              </div>

              {result && (
                <ResultCard
                  result={result}
                  compact={false}
                  predictionClass={predictionClass}
                  toneClass={getModelToneClass(result.model_key || selectedModel)}
                />
              )}
            </section>
          )}

          {activeTab === "archivo" && (
            <section className="card">
              <h2>Clasificación por archivo</h2>
              <p className="helper-text">
                Sube un archivo CSV o Excel y clasifícalo usando el modelo activo.
              </p>

              {selectedModelInfo &&
                (selectedModelInfo.type === "beto_llm_ensemble" ||
                  selectedModelInfo.type === "beto_llm_cascade") && (
                  <div className="alert-banner soft">
                    <AlertTriangle size={18} />
                    <span>
                      Este modo usa LLM además de BETO. Puede tardar más y consumir
                      créditos de Hugging Face al clasificar archivos grandes.
                    </span>
                  </div>
                )}

              <div className="two-columns">
                <div>
                  <label>Archivo CSV o Excel</label>
                  <input
                    type="file"
                    accept=".csv,.xlsx"
                    onChange={(e) => {
                      setSelectedFile(e.target.files[0]);
                      setFileResultUrl("");
                    }}
                  />
                </div>

                <div>
                  <label>Columna de texto</label>
                  <input
                    type="text"
                    value={fileTextColumn}
                    onChange={(e) => setFileTextColumn(e.target.value)}
                    placeholder="Ejemplo: text, texto, tweet_text..."
                  />
                </div>
              </div>

              <p className="helper-text">
                Si lo dejas vacío, el backend intentará detectar la columna automáticamente.
              </p>

              <div className="actions-row">
                <button className="btn btn-primary" onClick={classifyFile}>
                  {loading ? "Clasificando archivo..." : "Clasificar archivo"}
                </button>
              </div>

              {fileResults && (
                <>
                  <h3>Resumen general</h3>

                  <div className="metrics-grid">
                    <Metric label="Filas totales" value={fileResults.total_rows} />
                    <Metric label="Filas válidas" value={fileResults.valid_rows} />
                    <Metric label="Filas descartadas" value={fileResults.dropped_rows} />
                    <Metric label="Columna texto" value={fileResults.text_column} />
                  </div>

                  <div className="table-wrap">
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
                  </div>

                  <h3>Distribución</h3>
                  <div className="chart-card">
                    <ResponsiveContainer width="100%" height={320}>
                      <BarChart data={fileResults.summary}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.18} />
                        <XAxis dataKey="clase_predicha" stroke="#b9c0d4" />
                        <YAxis stroke="#b9c0d4" />
                        <Tooltip />
                        <Bar dataKey="cantidad" radius={[10, 10, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <h3>Resultados detallados</h3>
                  <div className="table-wrap">
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

                  <p className="helper-text">
                    Mostrando las primeras 30 filas. Puedes descargar el CSV completo.
                  </p>
                </>
              )}

              {fileResultUrl && (
                <div className="download-box">
                  <p>Archivo clasificado correctamente.</p>
                  <a href={fileResultUrl} download={fileResultName} className="download-link">
                    <Download size={16} />
                    <span>Descargar resultados CSV</span>
                  </a>
                </div>
              )}
            </section>
          )}

          {activeTab === "comparacion" && (
            <section className="card">
              <h2>Comparación entre modelos</h2>
              <p className="helper-text">
                Evalúa el mismo texto con todos los modelos disponibles.
              </p>

              <label>Texto para comparar entre modelos</label>
              <textarea
                value={compareText}
                onChange={(e) => setCompareText(e.target.value)}
                placeholder="Ejemplo: llevo dos días ayunando porque me siento enorme..."
              />

              <div className="actions-row">
                <button className="btn btn-primary" onClick={compareModels}>
                  {loading ? "Comparando..." : "Comparar modelos"}
                </button>
              </div>

              {compareResults.length > 0 && (
                <>
                  <div className="compare-grid">
                    {compareResults.map((item, index) => (
                      <ResultCard
                        key={index}
                        result={item}
                        compact
                        predictionClass={predictionClass}
                        toneClass={getModelToneClass(item.model_key)}
                      />
                    ))}
                  </div>

                  <h3>Resumen comparativo</h3>

                  <div className="table-wrap">
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
                  </div>

                  <h3>Gráfica comparativa</h3>
                  <div className="chart-card">
                    <ResponsiveContainer width="100%" height={320}>
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
                        <CartesianGrid strokeDasharray="3 3" opacity={0.18} />
                        <XAxis dataKey="modelo" stroke="#b9c0d4" />
                        <YAxis domain={[0, 1]} stroke="#b9c0d4" />
                        <Tooltip />
                        <Bar dataKey="probabilidad" radius={[10, 10, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </>
              )}
            </section>
          )}
        </main>
      </div>
    </div>
  );
}

function ResultCard({ result, compact = false, predictionClass, toneClass = "" }) {
  return (
    <section className={`result-card ${toneClass}`}>
      <div className="result-card-header">
        <div>
          <h3>{result.short_label || result.model_label || "Resultado"}</h3>
          {(result.model_label || result.family) && (
            <p className="helper-text">
              {result.model_label}
              {result.family ? ` · ${result.family}` : ""}
            </p>
          )}
        </div>

        {result.model_key?.includes("ensemble") && (
          <span className="tiny-recommendation">Recomendado</span>
        )}
      </div>

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

      {compact ? (
        <details className="details-box">
          <summary>Ver texto procesado</summary>
          <pre>{result.cleaned_text}</pre>
        </details>
      ) : (
        <>
          <h4>Texto procesado</h4>
          <pre>{result.cleaned_text}</pre>
        </>
      )}
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric-card">
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