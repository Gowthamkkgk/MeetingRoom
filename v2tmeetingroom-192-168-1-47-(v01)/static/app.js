// static/app.js - OPTIMIZED FOR AUDIO QUALITY
let ws = null;
let meetingId = null;
let isConnected = false;
let isMeetingActive = false;
let isRecording = false;
let isEchoEnabled = false;
let audioContext = null;
let audioStream = null;
let isAudioRecording = false;
let userData = {
    id: null,
    name: "User"
};
let participants = [];
let transcriptionCheckInterval = null;
 
// Enhanced audio processing variables
let audioProcessors = new Map();
const AUDIO_BUFFER_SIZE = 2048; // Smaller buffer for lower latency
const SAMPLE_RATE = 16000;
const CHUNK_DURATION_MS = 50; // Send every 50ms for smoother streaming
 
// Audio quality monitoring
let audioQualityStats = {
    chunksSent: 0,
    chunksReceived: 0,
    validationFailed: 0,
    lastCleanup: Date.now()
};
 
// Check browser compatibility
function checkBrowserCompatibility() {
    const warningDiv = document.getElementById('browserWarning');
    const warningText = document.getElementById('browserWarningText');
    let issues = [];
 
    if (!navigator.mediaDevices) {
        issues.push('MediaDevices API not supported');
    } else if (!navigator.mediaDevices.getUserMedia) {
        issues.push('getUserMedia not supported');
    }
 
    if (!window.WebSocket) {
        issues.push('WebSocket not supported');
    }
 
    if (!window.AudioContext && !window.webkitAudioContext) {
        issues.push('Web Audio API not supported');
    }
 
    if (!window.isSecureContext) {
        issues.push('Not a secure context (HTTPS or localhost required)');
    }
 
    if (issues.length > 0) {
        warningText.innerHTML = `Your browser has compatibility issues:
            <ul>${issues.map(issue => `<li>${issue}</li>`).join('')}</ul>
            <strong>Recommended:</strong> Use Chrome, Firefox, or Edge on desktop with https://localhost:5050`;
        warningDiv.style.display = 'block';
        return false;
    }
    warningDiv.style.display = 'none';
    return true;
}
 
function log(message, type = 'info') {
    console.log(`[${type.toUpperCase()}]`, message);
    const transcriptDiv = document.getElementById('transcript');
    const timestamp = new Date().toLocaleTimeString();
    if (transcriptDiv) {
        const messageDiv = document.createElement('div');
        messageDiv.innerHTML = `<strong>[${timestamp}]</strong> ${message}`;
       
        switch(type) {
            case 'error':
                messageDiv.className = 'message-error';
                break;
            case 'success':
                messageDiv.className = 'message-success';
                break;
            case 'audio':
                messageDiv.className = 'message-audio';
                break;
            case 'system':
                messageDiv.className = 'message-system';
                break;
            case 'chat':
                messageDiv.className = 'message-chat';
                break;
            case 'echo':
                messageDiv.className = 'message-echo';
                break;
            case 'transcription':
                messageDiv.className = 'message-transcription';
                break;
            case 'quality':
                messageDiv.className = 'message-quality';
                break;
            default:
                messageDiv.className = 'message-system';
        }
       
        transcriptDiv.appendChild(messageDiv);
        transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
    }
}
 
function updateConnectionStatus(connected) {
    isConnected = connected;
    const statusElement = document.getElementById('connectionStatus');
    if (statusElement) {
        if (connected) {
            statusElement.innerHTML = '✅ Connected';
            statusElement.className = 'status-value status-connected';
        } else {
            statusElement.innerHTML = '❌ Disconnected';
            statusElement.className = 'status-value status-disconnected';
        }
    }
}
 
function updateMeetingStatus(active) {
    isMeetingActive = active;
    const meetingStatusElement = document.getElementById('meetingStatus');
    const startMeetingBtn = document.getElementById('startMeetingBtn');
    const stopMeetingBtn = document.getElementById('stopMeetingBtn');
   
    if (meetingStatusElement) {
        if (active) {
            meetingStatusElement.innerHTML = '✅ Active';
            meetingStatusElement.className = 'status-value status-active';
        } else {
            meetingStatusElement.innerHTML = '🛑 Inactive';
            meetingStatusElement.className = 'status-value status-inactive';
        }
    }
   
    if (startMeetingBtn) startMeetingBtn.disabled = active;
    if (stopMeetingBtn) stopMeetingBtn.disabled = !active;
}
 
function updateRecordingStatus(recording) {
    isRecording = recording;
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingInfo = document.getElementById('recordingInfo');
   
    if (recordingStatus) {
        if (recording) {
            recordingStatus.innerHTML = '🔴 Recording...';
            recordingStatus.className = 'status-value status-active';
        } else {
            recordingStatus.innerHTML = '⚫ Off';
            recordingStatus.className = 'status-value status-inactive';
        }
    }
   
    if (recordingInfo) {
        recordingInfo.style.display = recording ? 'block' : 'none';
    }
}
 
function updateEchoStatus(enabled) {
    isEchoEnabled = enabled;
    const echoStatus = document.getElementById('echoStatus');
    if (echoStatus) {
        if (enabled) {
            echoStatus.innerHTML = '🔊 Enabled';
            echoStatus.className = 'status-value status-active';
        } else {
            echoStatus.innerHTML = '🔇 Disabled';
            echoStatus.className = 'status-value status-inactive';
        }
    }
}
 
function updateAudioStatus(recording) {
    isAudioRecording = recording;
    const audioStatus = document.getElementById('audioStatus');
    const audioIndicator = document.querySelector('.audio-indicator');
    const startAudioBtn = document.getElementById('startAudioBtn');
    const stopAudioBtn = document.getElementById('stopAudioBtn');
   
    if (audioStatus) {
        if (recording) {
            audioStatus.innerHTML = '🎤 Recording';
            audioStatus.className = 'status-value status-active';
        } else {
            audioStatus.innerHTML = '🔴 Off';
            audioStatus.className = 'status-value status-inactive';
        }
    }
   
    if (audioIndicator) {
        if (recording) {
            audioIndicator.classList.add('recording');
        } else {
            audioIndicator.classList.remove('recording');
        }
    }
   
    if (startAudioBtn) startAudioBtn.disabled = recording;
    if (stopAudioBtn) stopAudioBtn.disabled = !recording;
}
 
function updateParticipantsList(participantsList) {
    participants = participantsList || [];
    const participantsListElement = document.getElementById('participantsList');
    const participantsCountElement = document.getElementById('participantsCount');
    const sidebarParticipantsCount = document.getElementById('sidebarParticipantsCount');
   
    if (participantsListElement) {
        participantsListElement.innerHTML = '';
       
        // Add current user first
        const currentUserItem = document.createElement('div');
        currentUserItem.className = 'participant-item';
        currentUserItem.innerHTML = `
            <div class="participant-avatar">${userData.name.charAt(0).toUpperCase()}</div>
            <span>You (${userData.name})</span>
            ${isAudioRecording ? '🎤' : ''}
        `;
        participantsListElement.appendChild(currentUserItem);
       
        // Add other participants
        participants.forEach(participant => {
            if (participant.user_id !== userData.id) {
                const participantItem = document.createElement('div');
                participantItem.className = `participant-item ${participant.is_speaking ? 'participant-speaking' : ''}`;
                participantItem.innerHTML = `
                    <div class="participant-avatar">${participant.user_name.charAt(0).toUpperCase()}</div>
                    <span>${participant.user_name}</span>
                    ${participant.is_speaking ? '🎤' : ''}
                `;
                participantsListElement.appendChild(participantItem);
            }
        });
    }
   
    const totalParticipants = participants.length + 1;
    if (participantsCountElement) {
        participantsCountElement.textContent = totalParticipants;
    }
    if (sidebarParticipantsCount) {
        sidebarParticipantsCount.textContent = totalParticipants;
    }
}
 
function updateUserSpeakingStatus(userId, isSpeaking) {
    const participantsList = document.getElementById('participantsList');
    if (participantsList) {
        const participantItems = participantsList.getElementsByClassName('participant-item');
        for (let item of participantItems) {
            const nameElement = item.querySelector('span');
            if (nameElement && nameElement.textContent.includes(userId)) {
                if (isSpeaking) {
                    item.classList.add('participant-speaking');
                    if (!item.innerHTML.includes('🎤')) {
                        item.innerHTML += ' 🎤';
                    }
                } else {
                    item.classList.remove('participant-speaking');
                    item.innerHTML = item.innerHTML.replace(' 🎤', '');
                }
                break;
            }
        }
    }
}
 
// Enhanced audio playback with better quality
function playAudioData(audioData, userId, userName) {
    if (!audioData || !audioData.length) {
        return;
    }
   
    try {
        // Create audio context if needed
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: SAMPLE_RATE,
                latencyHint: 'interactive'
            });
        }
 
        // Validate audio data before processing
        if (!validateAudioData(audioData)) {
            audioQualityStats.validationFailed++;
            return;
        }
 
        // Get or create audio processor for this user
        let processor = audioProcessors.get(userId);
        if (!processor) {
            processor = {
                buffers: [],
                isPlaying: false,
                lastPlayTime: 0,
                playbackGap: 0
            };
            audioProcessors.set(userId, processor);
        }
 
        // Add new audio data to buffer with timestamp
        const audioBuffer = new Float32Array(audioData);
        processor.buffers.push({
            data: audioBuffer,
            timestamp: Date.now()
        });
 
        // Clean old buffers to prevent memory leaks
        cleanupOldBuffers();
 
        // If not currently playing, start playback
        if (!processor.isPlaying) {
            processAudioBuffer(userId, userName);
        }
 
        audioQualityStats.chunksReceived++;
       
        // Update speaking status
        updateUserSpeakingStatus(userName, true);
       
    } catch (error) {
        console.error('Audio playback error:', error);
    }
}
 
// Audio validation for incoming data
function validateAudioData(audioData) {
    if (!audioData || !Array.isArray(audioData)) return false;
    if (audioData.length === 0 || audioData.length > 10000) return false;
   
    // Check for reasonable values
    let sum = 0;
    let max = 0;
    for (let i = 0; i < Math.min(audioData.length, 100); i++) {
        const val = Math.abs(audioData[i]);
        sum += val;
        if (val > max) max = val;
    }
   
    const avg = sum / Math.min(audioData.length, 100);
    return avg < 10.0 && max < 100.0; // Reasonable amplitude limits
}
 
// Cleanup old audio buffers
function cleanupOldBuffers() {
    const now = Date.now();
    if (now - audioQualityStats.lastCleanup < 10000) return; // Cleanup every 10 seconds
   
    audioQualityStats.lastCleanup = now;
   
    for (let [userId, processor] of audioProcessors) {
        // Remove buffers older than 5 seconds
        processor.buffers = processor.buffers.filter(buffer =>
            now - buffer.timestamp < 5000
        );
       
        // Remove processor if no buffers and not playing
        if (processor.buffers.length === 0 && !processor.isPlaying) {
            audioProcessors.delete(userId);
        }
    }
}
 
function processAudioBuffer(userId, userName) {
    const processor = audioProcessors.get(userId);
    if (!processor || processor.buffers.length === 0) {
        processor.isPlaying = false;
        // Delay clearing speaking status to avoid flickering
        setTimeout(() => updateUserSpeakingStatus(userName, false), 500);
        return;
    }

    processor.isPlaying = true;

    try {
        // Get the next buffer
        const bufferObj = processor.buffers.shift();
        const audioBuffer = bufferObj.data;

        // Calculate optimal playback timing
        const now = Date.now();
        const timeSinceLastPlay = now - processor.lastPlayTime;
        const bufferDuration = (audioBuffer.length / SAMPLE_RATE) * 1000;

        // Create Web Audio buffer
        const buffer = audioContext.createBuffer(1, audioBuffer.length, SAMPLE_RATE);
        buffer.getChannelData(0).set(audioBuffer);

        // Create and play audio
        const source = audioContext.createBufferSource();
        source.buffer = buffer;

        // Add gain control for consistent volume
        const gainNode = audioContext.createGain();
        gainNode.gain.value = 1.3; // Increased gain for clearer output
        source.connect(gainNode);
        gainNode.connect(audioContext.destination);

        // Schedule playback with timing correction
        const playTime = Math.max(0, (processor.lastPlayTime + bufferDuration) - now);
        source.start(audioContext.currentTime + (playTime / 1000));

        processor.lastPlayTime = now + playTime;

        // Process next buffer
        const nextBufferDelay = Math.max(bufferDuration - 10, 10); // Slightly faster than buffer duration
        setTimeout(() => {
            processAudioBuffer(userId, userName);
        }, nextBufferDelay);

    } catch (error) {
        console.error('Audio processing error:', error);
        // Retry with delay on error
        setTimeout(() => {
            processAudioBuffer(userId, userName);
        }, 10);
    }
}
 
// Audio echo playback with quality improvements
function playAudioEcho(audioData, originalSpeaker) {
    if (!audioData || !audioData.length) return;
   
    try {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: SAMPLE_RATE
            });
        }
 
        // Validate echo audio
        if (!validateAudioData(audioData)) return;
 
        // Create Web Audio buffer for echo
        const buffer = audioContext.createBuffer(1, audioData.length, SAMPLE_RATE);
        buffer.getChannelData(0).set(new Float32Array(audioData));
       
        // Create and play audio with lower volume for echo
        const source = audioContext.createBufferSource();
        source.buffer = buffer;
       
        const gainNode = audioContext.createGain();
        gainNode.gain.value = 0.4; // Slightly higher volume for better clarity
       
        source.connect(gainNode);
        gainNode.connect(audioContext.destination);
       
        source.start(0);
       
        log(`🔊 Context from ${originalSpeaker}`, 'echo');
       
    } catch (error) {
        console.error('Audio echo playback error:', error);
    }
}
 
// Enhanced audio recording with better quality
async function startAudio() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        log('❌ Not connected to meeting', 'error');
        return;
    }
   
    if (!isMeetingActive) {
        log('❌ Meeting is not active. Please start the meeting first.', 'error');
        return;
    }
 
    try {
        log('🎤 Starting microphone with high quality settings...', 'system');
       
        // Get high-quality microphone access with optimal settings
        audioStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: { ideal: true },
                noiseSuppression: { ideal: true },
                autoGainControl: { ideal: false }, // Disable auto gain for better control
                channelCount: 1,
                sampleRate: SAMPLE_RATE,
                sampleSize: 16,
                latency: 0.01
            },
            video: false
        });
 
        log('✅ Microphone access granted! Configuring audio processing...', 'success');
 
        // Create audio context with optimal settings
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: SAMPLE_RATE,
                latencyHint: 'interactive'
            });
        }
 
        // Wait for audio context to be ready
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }
 
        // Create audio source from stream
        const source = audioContext.createMediaStreamSource(audioStream);
       
        // Create processor for continuous audio capture
        const processor = audioContext.createScriptProcessor(AUDIO_BUFFER_SIZE, 1, 1);
       
        let lastSendTime = 0;
        let silenceFrames = 0;
        const SILENCE_THRESHOLD = 0.01; // Lower threshold for better sensitivity
       
        processor.onaudioprocess = (event) => {
            if (!isAudioRecording || !isMeetingActive) return;
           
            const currentTime = Date.now();
            if (currentTime - lastSendTime < CHUNK_DURATION_MS) return;
           
            // Get audio data
            const inputData = event.inputBuffer.getChannelData(0);
           
            // Process audio for better quality
            const processedAudio = processAudioChunk(inputData);
           
            // Check if audio is active (not silence)
            const isActive = isAudioActive(processedAudio, SILENCE_THRESHOLD);
           
            if (isActive) {
                silenceFrames = 0;
               
                // Convert to array for JSON
                const audioArray = Array.from(processedAudio);
               
                // Validate before sending
                if (validateAudioData(audioArray)) {
                    // Send audio data
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'audio_data',
                            data: audioArray,
                            meeting_id: meetingId,
                            timestamp: new Date().toISOString()
                        }));
                       
                        audioQualityStats.chunksSent++;
                    }
                } else {
                    audioQualityStats.validationFailed++;
                }
               
                lastSendTime = currentTime;
            } else {
                silenceFrames++;
                // After 10 seconds of silence, send a stop speaking event
                if (silenceFrames > 200) { // ~10 seconds
                    silenceFrames = 0;
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'stop_speaking',
                            meeting_id: meetingId,
                            timestamp: new Date().toISOString()
                        }));
                    }
                }
            }
        };
 
        // Connect processing chain
        source.connect(processor);
        processor.connect(audioContext.destination);
       
        updateAudioStatus(true);
        log('🔊 Microphone active - you can speak now', 'success');
        log('💡 Speaking indicator will show when others are talking', 'system');
 
    } catch (error) {
        const errorMsg = `❌ Failed to start microphone: ${error.message}`;
        log(errorMsg, 'error');
       
        if (error.name === 'NotAllowedError') {
            log('💡 Click "Fix Microphone" button to grant permission', 'info');
        } else if (error.name === 'NotFoundError') {
            log('💡 No microphone found. Please check your audio devices.', 'error');
        } else if (error.name === 'NotSupportedError') {
            log('💡 Your browser doesn\'t support the required audio features.', 'error');
        }
       
        console.error('Audio error details:', error);
    }
}
 
// Audio processing for better quality
function processAudioChunk(inputData) {
    const output = new Float32Array(inputData.length);
    // Simple high-pass filter to remove DC offset
    let prev = 0;
    const alpha = 0.95;
    for (let i = 0; i < inputData.length; i++) {
        output[i] = alpha * (prev + inputData[i] - (i > 0 ? inputData[i-1] : 0));
        prev = output[i];
    }
    // Always normalize to target peak unless silent
    let max = 0;
    for (let i = 0; i < output.length; i++) {
        const absVal = Math.abs(output[i]);
        if (absVal > max) max = absVal;
    }
    const targetPeak = 0.9;
    if (max > 0.001) { // Only skip normalization if truly silent
        const factor = targetPeak / max;
        for (let i = 0; i < output.length; i++) {
            output[i] *= factor;
        }
    }
    return output;
}
 
// Improved audio activity detection
function isAudioActive(inputData, threshold = 0.01) {
    let sum = 0;
    for (let i = 0; i < inputData.length; i++) {
        sum += Math.abs(inputData[i]);
    }
    const average = sum / inputData.length;
    return average > threshold;
}
 
function stopAudio() {
    if (audioStream) {
        audioStream.getTracks().forEach(track => {
            track.stop();
            track.enabled = false;
        });
        audioStream = null;
    }
   
    updateAudioStatus(false);
    log('⏹️ Microphone stopped', 'warning');
   
    // Send stop speaking event
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'stop_speaking',
            meeting_id: meetingId,
            timestamp: new Date().toISOString()
        }));
    }
   
    // Log audio quality stats
    logAudioQualityStats();
}
 
// Log audio quality statistics
function logAudioQualityStats() {
    const totalChunks = audioQualityStats.chunksSent + audioQualityStats.validationFailed;
    if (totalChunks > 0) {
        const successRate = (audioQualityStats.chunksSent / totalChunks) * 100;
        log(`📊 Audio Quality: ${audioQualityStats.chunksSent}/${totalChunks} chunks sent (${successRate.toFixed(1)}% success rate)`, 'quality');
    }
   
    // Reset stats
    audioQualityStats = {
        chunksSent: 0,
        chunksReceived: 0,
        validationFailed: 0,
        lastCleanup: Date.now()
    };
}
 
// Microphone permission test
async function fixMicrophonePermissions() {
    log('🎤 Requesting microphone permission...', 'system');
   
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: false
            },
            video: false
        });
       
        stream.getTracks().forEach(track => track.stop());
       
        log('✅ Microphone permission granted! You can now use Mic On.', 'success');
        return true;
    } catch (error) {
        const errorMsg = `❌ Microphone permission denied: ${error.message}`;
        log(errorMsg, 'error');
       
        if (error.name === 'NotAllowedError') {
            log('💡 Please allow microphone access in your browser settings', 'info');
        }
       
        return false;
    }
}
 
function joinMeeting() {
    const userNameInput = document.getElementById('userName');
    const meetingIdInput = document.getElementById('meetingId');
   
    userData.name = userNameInput.value.trim() || 'User';
    meetingId = meetingIdInput.value.trim();
   
    if (!meetingId) {
        log('❌ Please enter a meeting ID', 'error');
        return;
    }
 
    log(`Attempting to join meeting: ${meetingId} as ${userData.name}`, 'system');
 
    // Close existing connection
    if (ws) {
        ws.close();
    }
 
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname;
    const port = window.location.port || '5050';
    const wsUrl = `${protocol}//${host}:${port}/ws/${meetingId}?user_name=${encodeURIComponent(userData.name)}`;
   
    log(`Connecting to: ${wsUrl}`, 'system');
    ws = new WebSocket(wsUrl);
 
    ws.onopen = () => {
        log('✅ WebSocket connected successfully!', 'success');
       
        document.getElementById('meetingControls').style.display = 'block';
        document.getElementById('meetingIdDisplay').style.display = 'block';
        document.getElementById('currentMeetingId').textContent = meetingId;
       
        updateConnectionStatus(true);
       
        // Reset audio quality stats
        logAudioQualityStats();
    };
 
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
           
            switch(data.type) {
                case 'audio_data':
                    playAudioData(data.data, data.user_id, data.user_name);
                    break;
                   
                case 'audio_echo':
                    playAudioEcho(data.data, data.original_speaker);
                    break;
                   
                case 'stop_speaking':
                    updateUserSpeakingStatus(data.user_name, false);
                    break;
                   
                case 'meeting_status':
                    updateMeetingStatus(data.status === 'started');
                    if (data.recording !== undefined) {
                        updateRecordingStatus(data.recording);
                    }
                    if (data.transcription_started) {
                        showTranscriptionInProgress();
                    }
                    log(data.text, data.status === 'started' ? 'success' : 'warning');
                    break;
                   
                case 'system':
                    log(data.text, 'system');
                    if (data.participant_list) {
                        updateParticipantsList(data.participant_list);
                    }
                    if (data.user_id) {
                        userData.id = data.user_id;
                    }
                    if (data.recording !== undefined) {
                        updateRecordingStatus(data.recording);
                    }
                    break;
                   
                case 'test_response':
                    log(data.text, 'success');
                    if (data.participant_list) {
                        updateParticipantsList(data.participant_list);
                    }
                    if (data.meeting_active !== undefined) {
                        updateMeetingStatus(data.meeting_active);
                    }
                    if (data.recording !== undefined) {
                        updateRecordingStatus(data.recording);
                    }
                    break;
                   
                case 'participant_update':
                    updateParticipantsList(data.participants);
                    break;
                   
                case 'chat':
                    addChatMessage(data.sender, data.text, data.timestamp);
                    log(`💬 ${data.sender}: ${data.text}`, 'chat');
                    break;
                   
                case 'raise_hand':
                    log(`✋ ${data.user_name} raised their hand`, 'system');
                    break;
                   
                case 'echo_status':
                    updateEchoStatus(data.enabled);
                    log(data.text, 'system');
                    break;
                   
                case 'transcription_status':
                    handleTranscriptionStatus(data.status);
                    break;
                   
                case 'error':
                    log(`❌ ${data.text}`, 'error');
                    break;
            }
        } catch (error) {
            log(`Error parsing message: ${error}`, 'error');
        }
    };
 
    ws.onerror = (error) => {
        log('❌ WebSocket connection error', 'error');
        updateConnectionStatus(false);
    };
 
    ws.onclose = (event) => {
        log(`⚠️ WebSocket connection closed`, 'warning');
        updateConnectionStatus(false);
        updateMeetingStatus(false);
        updateAudioStatus(false);
        updateRecordingStatus(false);
       
        document.getElementById('meetingControls').style.display = 'none';
        document.getElementById('meetingIdDisplay').style.display = 'none';
        document.getElementById('recordingInfo').style.display = 'none';
        document.getElementById('transcriptionStatus').style.display = 'none';
       
        // Cleanup audio processors
        audioProcessors.clear();
       
        // Clear transcription check interval
        if (transcriptionCheckInterval) {
            clearInterval(transcriptionCheckInterval);
            transcriptionCheckInterval = null;
        }
       
        // Log final audio stats
        logAudioQualityStats();
    };
}
 
// ... (rest of the functions remain the same as your original, but I'll include the key ones for completeness)
 
function createMeeting() {
    const userNameInput = document.getElementById('userName');
    userData.name = userNameInput.value.trim() || 'User';
   
    log('Creating new meeting...', 'system');
   
    fetch('/create-meeting', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            title: `Meeting with ${userData.name}`,
            host_name: userData.name
        })
    })
    .then(response => response.json())
    .then(data => {
        const newMeetingId = data.meeting_id;
        log(`✅ Meeting created with ID: ${newMeetingId}`, 'success');
       
        document.getElementById('meetingId').value = newMeetingId;
        meetingId = newMeetingId;
       
        setTimeout(joinMeeting, 500);
    })
    .catch(error => {
        log(`❌ Error creating meeting: ${error}`, 'error');
    });
}
 
function startMeeting() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        log('❌ Not connected to meeting', 'error');
        return;
    }
   
    ws.send(JSON.stringify({
        type: 'start_meeting',
        meeting_id: meetingId,
        timestamp: new Date().toISOString()
    }));
   
    log('🎯 Starting meeting...', 'system');
}
 
function stopMeetingWithTranscription() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        log('❌ Not connected to meeting', 'error');
        return;
    }
   
    ws.send(JSON.stringify({
        type: 'stop_meeting',
        meeting_id: meetingId,
        timestamp: new Date().toISOString()
    }));
   
    log('🛑 Stopping meeting and starting transcription...', 'warning');
    showTranscriptionInProgress();
   
    if (isAudioRecording) {
        stopAudio();
    }
   
    // Start checking transcription status
    startTranscriptionStatusCheck();
}
 
function showTranscriptionInProgress() {
    const transcriptionStatus = document.getElementById('transcriptionStatus');
    if (transcriptionStatus) {
        transcriptionStatus.style.display = 'block';
    }
}
 
function hideTranscriptionInProgress() {
    const transcriptionStatus = document.getElementById('transcriptionStatus');
    if (transcriptionStatus) {
        transcriptionStatus.style.display = 'none';
    }
}
 
function startTranscriptionStatusCheck() {
    // Clear existing interval
    if (transcriptionCheckInterval) {
        clearInterval(transcriptionCheckInterval);
    }
   
    // Check every 5 seconds
    transcriptionCheckInterval = setInterval(() => {
        checkTranscriptionStatus();
    }, 5000);
   
    // Also check immediately
    checkTranscriptionStatus();
}
 
async function checkTranscriptionStatus() {
    if (!meetingId) return;
   
    try {
        // Request transcription status via WebSocket
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'get_transcription_status',
                meeting_id: meetingId,
                timestamp: new Date().toISOString()
            }));
        }
       
        // Also check via API
        const response = await fetch(`/meeting/${meetingId}/transcription-status`);
        const data = await response.json();
       
        handleTranscriptionStatus(data.status);
       
    } catch (error) {
        log(`Error checking transcription status: ${error}`, 'error');
    }
}
 
function handleTranscriptionStatus(status) {
    if (status.is_processing) {
        showTranscriptionInProgress();
        log('🔄 Transcription in progress...', 'transcription');
    } else if (status.transcript_exists) {
        hideTranscriptionInProgress();
        showTranscriptionResults(status);
        log('✅ Transcription completed!', 'success');
       
        // Clear the check interval
        if (transcriptionCheckInterval) {
            clearInterval(transcriptionCheckInterval);
            transcriptionCheckInterval = null;
        }
    } else if (!status.is_processing && !status.transcript_exists) {
        // Transcription might have failed or not started
        log('❌ Transcription may have failed or not started', 'error');
    }
}
 
function showTranscriptionResults(status) {
    const transcriptionResults = document.getElementById('transcriptionResults');
    const transcriptionContent = document.getElementById('transcriptionContent');
   
    if (transcriptionResults && transcriptionContent) {
        transcriptionResults.style.display = 'block';
       
        let content = '';
       
        if (status.transcript_preview) {
            content += `<h4>📝 Transcript Preview</h4>`;
            content += `<div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px;">`;
            content += `<pre style="white-space: pre-wrap; font-family: inherit;">${status.transcript_preview}</pre>`;
            content += `</div>`;
        }
       
        if (status.mom_preview) {
            content += `<h4>📋 Meeting Minutes Preview</h4>`;
            content += `<div style="background: white; padding: 15px; border-radius: 5px;">`;
            content += `<pre style="white-space: pre-wrap; font-family: inherit;">${status.mom_preview}</pre>`;
            content += `</div>`;
        }
       
        transcriptionContent.innerHTML = content;
    }
}
 
// ... (rest of your original functions like downloadTranscript, testWebSocket, etc. remain the same)
 
// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Add your existing event listeners here
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendChatMessage();
            }
        });
    }
   
    const meetingIdInput = document.getElementById('meetingId');
    if (meetingIdInput) {
        meetingIdInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                joinMeeting();
            }
        });
    }
   
    const userNameInput = document.getElementById('userName');
    if (userNameInput) {
        userNameInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                joinMeeting();
            }
        });
    }
   
    log('🎯 Enhanced Meeting App initialized', 'system');
    log('🔊 Audio quality improvements enabled', 'quality');
   
    const isCompatible = checkBrowserCompatibility();
    if (isCompatible) {
        log('✅ Browser compatibility check passed', 'success');
        log('💡 Create a new meeting or join an existing one to start', 'system');
    }
 
    if (!window.location.hostname.includes('localhost') && !window.location.hostname.includes('127.0.0.1')) {
        log('⚠️ For best microphone access, use https://localhost:5050', 'warning');
    }
});
 
// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (ws) {
        ws.close();
    }
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
    }
    if (audioContext) {
        audioContext.close();
    }
    if (transcriptionCheckInterval) {
        clearInterval(transcriptionCheckInterval);
    }
   
    logAudioQualityStats();
});
