import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [prediction, setPrediction] = useState("");
  const [textInput, setTextInput] = useState("");

  const sendDummyLandmarks = () => {
    const dummy = Array(30).fill(Array(42).fill(0.5));  // Simulated data
    axios.post("http://localhost:5000/predict", { landmarks: dummy })
      .then(res => setPrediction(res.data.label))
      .catch(err => console.error("Error:", err));
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Sign Language Translator</h1>

      <button onClick={sendDummyLandmarks}>Test Sign → Text</button>
      <h2>Prediction: {prediction}</h2>

      <hr />

      <input
        value={textInput}
        onChange={(e) => setTextInput(e.target.value)}
        placeholder="Enter word to view gesture"
      />
      <video
        width="300"
        controls
        src={`http://localhost:5000/video/${textInput}.mp4`}
      />
    </div>
  );
}

export default App;
