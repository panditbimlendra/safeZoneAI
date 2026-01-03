const express = require('express');
const path = require('path');
const app = express();
const port = process.env.PORT || 3000;

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// API endpoint for video analysis
app.post('/api/analyze-crash', (req, res) => {
  // In a real implementation, this would process the video
  // For demo, we return a simulated response
  const crashDetected = Math.random() > 0.3;
  
  if (crashDetected) {
    // Return crash image
    res.json({
      result: 'crash',
      image: 'https://images.unsplash.com/photo-1503376780353-7e6692767b70?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80'
    });
  } else {
    res.json({ result: 'no_crash' });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});