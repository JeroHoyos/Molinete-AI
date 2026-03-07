const { useState, useRef, useEffect, useCallback } = React;

function App() {
  const [input, setInput]       = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading]   = useState(false);
  const [history, setHistory]   = useState([]);
  const [cardVisible, setCardVisible] = useState(true);
  const [bgPos, setBgPos]       = useState({ x: 50, y: 50 });

  const inputRef   = useRef(null);
  const cardRef    = useRef(null);
  const rafRef     = useRef(null);
  const targetPos  = useRef({ x: 50, y: 50 });
  const currentPos = useRef({ x: 50, y: 50 });

  // Smooth parallax animation loop
  useEffect(() => {
    function animate() {
      const cp = currentPos.current;
      const tp = targetPos.current;
      const ease = 0.045;
      cp.x += (tp.x - cp.x) * ease;
      cp.y += (tp.y - cp.y) * ease;
      setBgPos({ x: cp.x, y: cp.y });
      rafRef.current = requestAnimationFrame(animate);
    }
    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  const handleMouseMove = useCallback((e) => {
    const { innerWidth: W, innerHeight: H } = window;
    // Map mouse to 44–56% range for subtle parallax drift
    targetPos.current = {
      x: 44 + (e.clientX / W) * 12,
      y: 44 + (e.clientY / H) * 12,
    };

    // Show card only when cursor is near it
    const card = cardRef.current;
    if (!card) return;
    const rect = card.getBoundingClientRect();
    const overCard =
      e.clientX >= rect.left - 24 && e.clientX <= rect.right  + 24 &&
      e.clientY >= rect.top  - 24 && e.clientY <= rect.bottom + 24;
    setCardVisible(overCard);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setCardVisible(true);
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    const msg = input.trim();
    if (!msg) return;
    setLoading(true);
    setResponse(null);
    const form = new FormData();
    form.append('message', msg);
    try {
      const res  = await fetch('/echo', { method: 'POST', body: form });
      const data = await res.json();
      setResponse(data.response);
      setHistory(prev => [{ original: data.original, response: data.response }, ...prev].slice(0, 5));
    } catch (err) {
      setResponse('Error connecting to server.');
    } finally {
      setLoading(false);
      setInput('');
      inputRef.current?.focus();
    }
  }

  const bgStyle = { backgroundPosition: `${bgPos.x}% ${bgPos.y}%` };

  return (
    <div
      className="root-wrapper"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      <div className="bg" style={bgStyle} />
      <div className="bg-overlay" style={bgStyle} />

      <div className={`bg-hint ${!cardVisible ? 'bg-hint--visible' : ''}`}>
        move to the center to write
      </div>

      <div className="scene">
        <div
          ref={cardRef}
          className={`card ${cardVisible ? 'card--visible' : 'card--hidden'}`}
        >
          <p className="eyebrow">Molinete AI</p>
          <h1>send the word <br/><span> of destiny</span></h1>

          <form onSubmit={handleSubmit}>
            <div className="field">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder="Type anything…"
                disabled={loading}
                autoFocus
              />
              <button className="send-btn" type="submit" disabled={loading || !input.trim()}>
                Send →
              </button>
            </div>
          </form>

          <div className="divider" />

          <div className="response-area">
            <p className="label">Server Response</p>
            {loading && (
              <div className="loading-dots">
                <span /><span /><span />
              </div>
            )}
            {!loading && response && (
              <div className="bubble">
                <em className="tag">received + echo</em>
                {response}
              </div>
            )}
            {!loading && !response && (
              <p className="idle-hint">Waiting for your message…</p>
            )}
          </div>

          {history.length > 1 && (
            <div className="history">
              <p className="label" style={{marginBottom: 0}}>Recent</p>
              {history.slice(1).map((item, i) => (
                <div className="history-item" key={i}>
                  <strong>→</strong> {item.response}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById('root'));