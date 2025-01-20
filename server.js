// server.js
const express = require('express');
const multer = require('multer');
const path = require('path');
const fetch = require('node-fetch-commonjs');
const FormData = require('form-data');
const app = express();
const port = 3000;

// Configure multer for audio file uploads
const storage = multer.memoryStorage(); // Store file in memory for forwarding
const upload = multer({
    storage: storage,
    fileFilter: function(req, file, cb) {
        if (file.mimetype.startsWith('audio/')) {
            cb(null, true);
        } else {
            cb(new Error('Only audio files are allowed'));
        }
    }
});

// Serve static files
app.use(express.static('public'));

// Handle file upload and forward to Python server
app.post('/upload', upload.single('audio'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No audio file uploaded' });
        }

        // Create form data for Python server
        const formData = new FormData();
        formData.append('audio', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        // Forward to Python server
        const pythonResponse = await fetch('http://localhost:5000/process-audio', {
            method: 'POST',
            body: formData
        });

        const result = await pythonResponse.json();
        res.json(result);

    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to process audio'
        });
    }
});

app.listen(port, () => {
    console.log(`Node.js server running at http://localhost:${port}`);
});