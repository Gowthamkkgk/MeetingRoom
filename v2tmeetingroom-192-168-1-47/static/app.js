// static/app.js
let ws = null;
let mediaRecorder = null;
let meetingId = null;
let isConnected = false;
let audioStream = null;
let isRecording = false;

// Check browser compatibility
function checkBrowserCompatibility() {
    const warningDiv = document.getElementById('browserWarning');
    const warningText = document.getElementById('browserWarningText');
    let issues = [];

    // Check for MediaDevices support
    if (!navigator.mediaDevices) {
        issues.push('MediaDevices API not supported');
    } else if (!navigator.mediaDevices.getUserMedia) {
        issues.push('getUserMedia not supported');
    }

    // Check for WebSocket support
    if (!window.WebSocket) {
        issues.push('WebSocket not supported');
    }

    // Check for MediaRecorder support
    if (!window.MediaRecorder) {
        issues.push('MediaRecorder not supported');
    }

    // Check if we're on a secure context
    if (!window.isSecureContext) {
        issues.push('Not a secure context (HTTPS or localhost required)');
    }

    if (issues.length > 0) {
        warningText.innerHTML = `
            Your browser has compatibility issues:
            <ul>
                ${issues.map(issue => `<li>${issue}</li>`).join('')}
            </ul>
            <strong>Recommended:</strong> Use Chrome, Firefox, or Edge on desktop with http://localhost:5001
        `;
        warningDiv.style.display = 'block';
        return false;
    }

    warningDiv.style.display = 'none';
    return true;
}

// Polyfill for older browsers
function ensureMediaDevices() {
    if (!navigator.mediaDevices) {
        navigator.mediaDevices = {};
    }
    
    if (!navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia = function(constraints) {
            const getUserMedia = navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
            
            if (!getUserMedia) {
                return Promise.reject(new Error('getUserMedia is not implemented in this browser'));
            }
            
            return new Promise(function(resolve, reject) {
                getUserMedia.call(navigator, constraints, resolve, reject);
            });
        };
    }
}

function log(message, type = 'info') {
    console.log(message);
    const transcriptDiv = document.getElementById('transcript');
    const timestamp = new Date().toLocaleTimeString();
    
    if (transcriptDiv) {
        const messageDiv = document.createElement('div');
        messageDiv.innerHTML = `<strong>[${timestamp}]</strong> ${message}`;
        
        switch(type) {
            case 'error':
                messageDiv.style.color = '#dc3545';
                break;
            case 'success':
                messageDiv.style.color = '#28a745';
                break;
            case 'warning':
                messageDiv.style.color = '#ffc107';
                break;
            default:
                messageDiv.style.color = '#007bff';
        }
        
        transcriptDiv.appendChild(messageDiv);
        transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
    }
}

function showStatus(message, type = 'info') {
    const statusDiv = document.getElementById('status');
    statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
}

function updateConnectionStatus(connected) {
    isConnected = connected;
    const statusElement = document.getElementById('connectionStatus');
    if (connected) {
        statusElement.innerHTML = '✅ Connected';
        statusElement.style.color = '#28a745';
    } else {
        statusElement.innerHTML = '❌ Disconnected';
        statusElement.style.color = '#dc3545';
    }
}

function updateAudioStatus(recording) {
    isRecording = recording;
    const audioStatus = document.getElementById('audioStatus');
    const audioIndicator = document.querySelector('.audio-indicator');
    
    if (recording) {
        audioStatus.innerHTML = '🎤 Recording audio...';
        audioStatus.style.color = '#28a745';
        audioIndicator.classList.add('recording');
    } else {
        audioStatus.innerHTML = '🔴 Microphone not active';
        audioStatus.style.color = '#dc3545';
        audioIndicator.classList.remove('recording');
    }
}

function updateParticipantsCount(count) {
    const participantsElement = document.getElementById('participantsCount');
    const participantsInfo = document.getElementById('participantsInfo');
    
    if (participantsElement) {
        participantsElement.textContent = count;
    }
    if (participantsInfo && count > 0) {
        participantsInfo.style.display = 'block';
    }
}

// Simple microphone permission request
async function fixMicrophonePermissions() {
    log('🎤 Requesting microphone permission...');
    
    // Ensure mediaDevices is available
    ensureMediaDevices();
    
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const errorMsg = '❌ This browser does not support microphone access';
        log(errorMsg, 'error');
        showStatus(errorMsg, 'error');
        return false;
    }
    
    try {
        // Simple permission request with basic constraints
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: true, // Use basic audio constraints
            video: false
        });
        
        // Immediately stop the stream
        stream.getTracks().forEach(track => track.stop());
        
        log('✅ Microphone permission granted! You can now start audio.', 'success');
        showStatus('✅ Microphone permission granted!', 'success');
        return true;
        
    } catch (error) {
        const errorMsg = `❌ Microphone permission denied: ${error.message}`;
        log(errorMsg, 'error');
        showStatus('❌ Microphone access blocked. Please allow in browser settings.', 'error');
        
        if (error.name === 'NotAllowedError') {
            log('💡 Please allow microphone access in your browser settings:', 'info');
            log('1. Click the lock/camera icon in address bar', 'info');
            log('2. Select "Allow" for microphone', 'info');
            log('3. Refresh the page and try again', 'info');
        } else if (error.name === 'NotFoundError') {
            log('💡 No microphone detected on your device', 'info');
        } else if (error.name === 'NotSupportedError') {
            log('💡 Your browser does not support audio recording', 'info');
        }
        return false;
    }
}

function joinMeeting() {
    meetingId = document.getElementById('meetingId').value.trim();
    if (!meetingId) {
        showStatus('Please enter a meeting ID', 'error');
        return;
    }
    
    showStatus(`Joining meeting: ${meetingId}`, 'info');
    log(`Attempting to join meeting: ${meetingId}`);
    
    // Close existing connection
    if (ws) {
        ws.close();
    }
    
    // Use the correct WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname;
    const port = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
    const wsUrl = `${protocol}//${host}:5091/ws/${meetingId}`;
    
    log(`Connecting to: ${wsUrl}`);
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        showStatus('✅ Successfully joined meeting!', 'success');
        log('WebSocket connected successfully!', 'success');
        document.getElementById('meetingControls').style.display = 'block';
        document.getElementById('meetingIdDisplay').style.display = 'block';
        document.getElementById('currentMeetingId').textContent = meetingId;
        updateConnectionStatus(true);
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            
            if (data.type === 'transcript') {
                log(`🎯 ${data.text}`, 'success');
            } else if (data.type === 'test_response') {
                log(`✅ ${data.text}`, 'success');
                if (data.participants) {
                    updateParticipantsCount(data.participants);
                }
            } else if (data.type === 'system') {
                log(`🔔 ${data.text}`, 'info');
                if (data.text.includes('participants')) {
                    // Extract participant count from message
                    const match = data.text.match(/(\d+) participants/);
                    if (match) {
                        updateParticipantsCount(parseInt(match[1]));
                    }
                }
            } else if (data.type === 'confirmation') {
                log(`✅ ${data.text}`, 'success');
            } else if (data.type === 'participant_update') {
                updateParticipantsCount(data.count);
            }
            
        } catch (error) {
            log(`Error parsing message: ${error}`, 'error');
        }
    };
    
    ws.onerror = (error) => {
        showStatus('❌ Failed to connect to meeting', 'error');
        log('WebSocket connection error', 'error');
        console.error('WebSocket error:', error);
        updateConnectionStatus(false);
    };
    
    ws.onclose = (event) => {
        log(`WebSocket connection closed: ${event.code} - ${event.reason}`, 'warning');
        updateConnectionStatus(false);
        document.getElementById('meetingControls').style.display = 'none';
        document.getElementById('meetingIdDisplay').style.display = 'none';
        document.getElementById('participantsInfo').style.display = 'none';
    };
}

function createMeeting() {
    showStatus('Creating new meeting...', 'info');
    log('Creating new meeting...');
    
    fetch('/create-meeting')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            const newMeetingId = data.meeting_id;
            showStatus(`✅ Meeting created with ID: ${newMeetingId}`, 'success');
            log(`Meeting created with ID: ${newMeetingId}`, 'success');
            document.getElementById('meetingId').value = newMeetingId;
            meetingId = newMeetingId;
            
            // Auto-join after creation
            setTimeout(joinMeeting, 500);
        })
        .catch(error => {
            showStatus('❌ Error creating meeting', 'error');
            log(`Error creating meeting: ${error}`, 'error');
            console.error('Error:', error);
        });
}

async function startAudio() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        showStatus('❌ Not connected to meeting', 'error');
        return;
    }
    
    // Ensure mediaDevices is available
    ensureMediaDevices();
    
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const errorMsg = '❌ This browser does not support microphone access';
        log(errorMsg, 'error');
        showStatus(errorMsg, 'error');
        return;
    }
    
    try {
        log('🎤 Starting audio recording...');
        
        // Use basic audio constraints for maximum compatibility
        const constraints = {
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                sampleRate: 16000
            },
            video: false
        };
        
        audioStream = await navigator.mediaDevices.getUserMedia(constraints);
        log('✅ Microphone access granted!', 'success');
        
        // Check if MediaRecorder is supported
        if (typeof MediaRecorder === 'undefined') {
            throw new Error('MediaRecorder not supported in this browser');
        }
        
        // Setup media recorder with basic options
        const options = { mimeType: 'audio/webm' };
        mediaRecorder = new MediaRecorder(audioStream, options);
        
        mediaRecorder.ondataavailable = (event) => {
            if (ws && ws.readyState === WebSocket.OPEN && event.data.size > 0) {
                const reader = new FileReader();
                reader.onload = () => {
                    const base64Data = reader.result.split(',')[1];
                    ws.send(JSON.stringify({
                        type: 'audio_chunk',
                        data: base64Data,
                        meeting_id: meetingId,
                        timestamp: new Date().toISOString()
                    }));
                };
                reader.onerror = (error) => {
                    log('❌ Error reading audio data: ' + error, 'error');
                };
                reader.readAsDataURL(event.data);
            }
        };
        
        mediaRecorder.onerror = (event) => {
            log(`❌ Recording error: ${event.error}`, 'error');
            stopAudio();
        };
        
        mediaRecorder.onstop = () => {
            log('⏹️ MediaRecorder stopped', 'info');
        };
        
        // Start recording with 1-second chunks
        mediaRecorder.start(1000);
        updateAudioStatus(true);
        showStatus('🎤 Audio recording started!', 'success');
        
        // Update button states
        document.getElementById('startAudioBtn').disabled = true;
        document.getElementById('stopAudioBtn').disabled = false;
        
    } catch (error) {
        const errorMsg = `❌ Failed to start audio: ${error.message}`;
        log(errorMsg, 'error');
        showStatus('❌ Failed to start audio recording', 'error');
        
        if (error.name === 'NotAllowedError') {
            log('💡 Click "Fix Microphone" button to grant permission', 'info');
        } else if (error.name === 'NotFoundError') {
            log('💡 No microphone found on your device', 'info');
        } else if (error.name === 'NotSupportedError') {
            log('💡 Your browser does not support audio recording', 'info');
        }
        
        console.error('Audio error details:', error);
    }
}

function stopAudio() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        log('⏹️ Recording stopped', 'warning');
        updateAudioStatus(false);
        
        // Stop all tracks
        if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
            audioStream = null;
        }
        
        // Update button states
        document.getElementById('startAudioBtn').disabled = false;
        document.getElementById('stopAudioBtn').disabled = true;
    }
}

function testWebSocket() {
    if (!meetingId) {
        showStatus('Please create or join a meeting first', 'error');
        return;
    }
    
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'test',
            message: 'Connection test from client',
            timestamp: new Date().toISOString()
        }));
        log('✅ Test message sent to server');
    } else {
        log('❌ No active WebSocket connection', 'error');
    }
}

function clearTranscript() {
    const transcriptDiv = document.getElementById('transcript');
    if (transcriptDiv) {
        transcriptDiv.innerHTML = '';
        log('Transcript cleared', 'info');
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    log('🎯 Meeting App initialized');
    
    // Check browser compatibility
    const isCompatible = checkBrowserCompatibility();
    
    if (isCompatible) {
        log('✅ Browser compatibility check passed');
        log('💡 Create a new meeting or join an existing one to start');
        log('🔒 Click "Fix Microphone" if you have permission issues');
    } else {
        log('❌ Browser compatibility issues detected', 'error');
    }
    
    // Check if we're on localhost
    if (!window.location.hostname.includes('localhost') && 
        !window.location.hostname.includes('127.0.0.1')) {
        log('⚠️ For best microphone access, use http://localhost:5091', 'warning');
    }
    
    // Log current URL for debugging
    log(`🌐 Current URL: ${window.location.href}`);
});