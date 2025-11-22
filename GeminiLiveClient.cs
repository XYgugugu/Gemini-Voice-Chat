using UnityEngine;
using UnityEngine.InputSystem; // REQUIRED: New Input System Namespace
using System;
using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Threading;
using System.Threading.Tasks;

[RequireComponent(typeof(AudioSource))]
public class GeminiLiveClient : MonoBehaviour
{
    [Header("Configuration")]
    // Change localhost to 127.0.0.1 to avoid DNS lag
    public string serverUrl = "ws://127.0.0.1:8000/ws/chat";

    // CHANGED: KeyCode (Legacy) -> Key (New Input System)
    public Key pushToTalkKey = Key.Enter;

    [Header("Debug")]
    public bool isRecording = false;
    public string connectionStatus = "Disconnected";

    // Audio Config
    private const int SERVER_INPUT_RATE = 16000; // Gemini expects 16kHz
    private const int SERVER_OUTPUT_RATE = 24000; // Gemini sends 24kHz

    // Internal State
    private ClientWebSocket _ws;
    private CancellationTokenSource _cts;

    // Microphone
    private AudioClip _micClip;
    private string _micDevice;
    private int _lastMicPos = 0;

    // Audio Output Buffer (Thread-Safe)
    private ConcurrentQueue<float> _playbackQueue = new ConcurrentQueue<float>();

    async void Start()
    {
        // 1. Setup Speaker (Consumer)
        var source = GetComponent<AudioSource>();
        source.loop = true;
        source.clip = AudioClip.Create("SilentLoop", 44100, 1, 44100, false);
        source.Play();

        // 2. Setup Microphone (Producer)
        if (Microphone.devices.Length > 0)
        {
            _micDevice = Microphone.devices[0];
            // Ask for 16kHz.
            _micClip = Microphone.Start(_micDevice, true, 10, SERVER_INPUT_RATE);
            Debug.Log($"[Mic] Started on {_micDevice} at {SERVER_INPUT_RATE}Hz");
        }
        else
        {
            Debug.LogError("[Mic] No microphone found!");
            return;
        }

        // 3. Connect to WebSocket
        await ConnectToServer();
    }

    private async Task ConnectToServer()
    {
        _ws = new ClientWebSocket();
        _cts = new CancellationTokenSource();

        try
        {
            connectionStatus = "Connecting...";
            await _ws.ConnectAsync(new Uri(serverUrl), _cts.Token);
            connectionStatus = "Connected";
            Debug.Log("Connected to Gemini Live!");

            _ = ReceiveLoop();
        }
        catch (Exception e)
        {
            connectionStatus = "Error";
            Debug.LogError($"[WS] Connection Failed: {e.Message}");
        }
    }

    void Update()
    {
        // Toggle Recording - UPDATED for New Input System
        if (Keyboard.current != null && Keyboard.current[pushToTalkKey].wasPressedThisFrame)
        {
            isRecording = !isRecording;
            Debug.Log(isRecording ? "RECORDING..." : "PAUSED");
        }

        // Process Microphone Data on Main Thread
        if (_ws != null && _ws.State == WebSocketState.Open)
        {
            ProcessMicrophone();
        }
    }

    // --- TASK 1: Microphone (Producer) ---
    void ProcessMicrophone()
    {
        int currentPos = Microphone.GetPosition(_micDevice);
        if (currentPos < 0 || _lastMicPos == currentPos) return;

        int samplesToRead = 0;
        if (currentPos < _lastMicPos)
            samplesToRead = (_micClip.samples - _lastMicPos) + currentPos;
        else
            samplesToRead = currentPos - _lastMicPos;

        if (samplesToRead <= 0) return;

        float[] micData = new float[samplesToRead];

        if (currentPos < _lastMicPos)
        {
            // FIX: Handle Wrap-around safely
            int samplesToEnd = _micClip.samples - _lastMicPos;

            // Only read end part if we aren't exactly at the end
            if (samplesToEnd > 0)
            {
                float[] endPart = new float[samplesToEnd];
                _micClip.GetData(endPart, _lastMicPos);
                endPart.CopyTo(micData, 0);
            }

            // Read start part
            if (currentPos > 0)
            {
                float[] startPart = new float[currentPos];
                _micClip.GetData(startPart, 0);
                startPart.CopyTo(micData, samplesToEnd);
            }
        }
        else
        {
            _micClip.GetData(micData, _lastMicPos);
        }

        _lastMicPos = currentPos;

        if (isRecording)
        {
            byte[] pcm16 = FloatsToPCM16(micData);
            SendWebSocketMessage(pcm16);
        }
    }

    async void SendWebSocketMessage(byte[] data)
    {
        if (_ws.State != WebSocketState.Open) return;
        try
        {
            await _ws.SendAsync(new ArraySegment<byte>(data), WebSocketMessageType.Binary, true, _cts.Token);
        }
        catch (Exception e) { Debug.LogError($"[Send] Error: {e.Message}"); }
    }

    // --- TASK 2: Network Receiver (Network Thread) ---
    async Task ReceiveLoop()
    {
        var buffer = new byte[1024 * 4];

        try
        {
            while (_ws.State == WebSocketState.Open && !_cts.IsCancellationRequested)
            {
                var result = await _ws.ReceiveAsync(new ArraySegment<byte>(buffer), _cts.Token);

                if (result.MessageType == WebSocketMessageType.Close)
                {
                    connectionStatus = "Server Closed";
                    break;
                }

                int byteCount = result.Count;
                short[] pcm16 = new short[byteCount / 2];
                Buffer.BlockCopy(buffer, 0, pcm16, 0, byteCount);

                float[] audioChunk = new float[pcm16.Length];
                for (int i = 0; i < pcm16.Length; i++)
                {
                    audioChunk[i] = pcm16[i] / 32768f;
                }

                int unityRate = AudioSettings.outputSampleRate;
                float[] resampledChunk = Resample(audioChunk, SERVER_OUTPUT_RATE, unityRate);

                foreach (var sample in resampledChunk)
                {
                    _playbackQueue.Enqueue(sample);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[Receive] Error: {e.Message}");
        }
    }

    // --- TASK 3: Audio Output (Audio Thread) ---
    void OnAudioFilterRead(float[] data, int channels)
    {
        for (int i = 0; i < data.Length; i += channels)
        {
            if (_playbackQueue.TryDequeue(out float sample))
            {
                for (int c = 0; c < channels; c++)
                {
                    data[i + c] = sample;
                }
            }
            else
            {
                for (int c = 0; c < channels; c++)
                {
                    data[i + c] = 0f;
                }
            }
        }
    }

    // --- HELPERS ---
    private byte[] FloatsToPCM16(float[] samples)
    {
        short[] pcm16 = new short[samples.Length];
        for (int i = 0; i < samples.Length; i++)
        {
            float sample = Mathf.Clamp(samples[i], -1f, 1f);
            pcm16[i] = (short)(sample * 32767f);
        }

        byte[] bytes = new byte[samples.Length * 2];
        Buffer.BlockCopy(pcm16, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private float[] Resample(float[] src, int srcRate, int dstRate)
    {
        if (srcRate == dstRate) return src;

        float ratio = (float)srcRate / dstRate;
        int newLen = (int)(src.Length / ratio);
        float[] dst = new float[newLen];

        for (int i = 0; i < newLen; i++)
        {
            float srcIndex = i * ratio;
            int index0 = (int)srcIndex;
            int index1 = Mathf.Min(index0 + 1, src.Length - 1);
            float t = srcIndex - index0;
            dst[i] = (src[index0] * (1f - t)) + (src[index1] * t);
        }
        return dst;
    }

    private void OnDestroy()
    {
        _cts?.Cancel();
        _ws?.Dispose();
    }
}