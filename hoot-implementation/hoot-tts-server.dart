// HOOT implementation of csm streaming

// ignore_for_file: depend_on_referenced_packages
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:logger/logger.dart';
import 'package:get/get.dart';
import 'package:hoot/app/controllers/service_keys_controller.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:flutter_sound/flutter_sound.dart';

class GoogleTTSController extends GetxController {
  final FlutterSoundPlayer _player = FlutterSoundPlayer();
  WebSocketChannel? _channel;
  StreamSubscription? _channelSub;

  final keys = Get.put(ServiceKeysController());
  RxBool isLoading = false.obs;
  RxBool isSpeaking = false.obs;
  RxString filePath = "".obs;
  Function? onCompletionHandler;

  final List<Uint8List> _receivedChunks =
      []; // only used to keep track of length
  final String _wsUrl = "ws://hive.hussainkazarani.site/ws";

  @override
  void onInit() {
    super.onInit();
    Logger().d("TTS Init (streaming step1) 🔊");
    // Start initialization
    _initPlayerAndWs();
  }

  Future<void> _initPlayerAndWs() async {
    // await _player.stopPlayer(); // invalidate old player
    // await _player.openPlayer();
    // await _player.startPlayerFromStream(
    //   codec: Codec.pcm16,
    //   numChannels: 1,
    //   sampleRate: 24000,
    //   interleaved: false,
    //   bufferSize: 1024,
    // );

    Logger().d("FlutterSoundPlayer opened 🪽");
    _connectWebSocket();
  }

  void _connectWebSocket() {
    // cancel/close any existing stream sinks/ws channels
    _channelSub?.cancel();
    _channel?.sink.close();

    _channel = WebSocketChannel.connect(Uri.parse(_wsUrl));

    // Listen to incoming websocket messages
    _channelSub = _channel!.stream.listen(
      (event) => _handleWsMessage(event),
      onError: (err) {
        Logger().e("❌ WebSocket error: $err");
      },
      onDone: () {
        Logger().d("👋 WebSocket closed by server / done");
      },
      cancelOnError: true,
    );

    Logger().d("🌐 Connected to WS: $_wsUrl");
  }

  // event is json format response
  Future<void> _handleWsMessage(dynamic event) async {
    final Map<String, dynamic> data = jsonDecode(event as String);
    final String? type = (data['type'] as String?)?.toLowerCase();

    if (type == null) {
      Logger().w("❌ WS message type is NULL: $data");
      return;
    }

    // Stop and reset player if it's already open
    if (_player.isOpen()) {
      await _player.stopPlayer();
      await _player.closePlayer();
    }
    _receivedChunks.clear();

    // Always reopen fresh player
    await _player.openPlayer();
    await _player.startPlayerFromStream(
      codec: Codec.pcm16,
      numChannels: 1,
      sampleRate: 24000,
      interleaved: false,
      bufferSize: 1024,
    );

    Logger().i("📩 WS EVENT: $data");

    switch (type) {
      case "audio_status":
        final status = (data["status"] ?? "").toLowerCase();
        if (status == "generating") {
          isLoading.value = true;
          isSpeaking.value = false;
        } else if (status == "complete") {
          //LATER
          Logger().d("👾 Stream successfully completed with status");
          isLoading.value = false;
          // Don't instantly stop speaking here — wait for the player to finish!
          // Let the playerStateStream flip isSpeaking = false when playback ends.
          // simple delay approach
          Future.delayed(Duration(milliseconds: 2000), () {
            isSpeaking.value = false;
          });
        }
        break;

      case "audio_chunk":
        // First chunk: mark speaking + stop loading
        // Server sends Float32 samples, as in your working prototype:
        // { "type": "audio_chunk", "audio": [0.0, 0.12, ...] }
        if (isLoading.value) {
          // await Future.delayed(const Duration(milliseconds: 300));
          isLoading.value = false;
          isSpeaking.value = true;
        }
        _handleAudioChunk(data["audio"]);
        // if (!_player.isPlaying) await startPlayer();
        break;

      case "completion":
        // Logger().d("👾 Stream successfully completed. Drain Player and stop");
        // Logger().d("👾 Stream successfully completed. Stopping player");
        // isSpeaking.value = false;
        // await _player.stopPlayer(); // stop the stream
        break;
    }
  }

  void _handleAudioChunk(List<dynamic> floatList) {
    final floats = Float32List.fromList(floatList.cast<double>());
    final int16 = Int16List(floats.length);

    for (int i = 0; i < floats.length; i++) {
      final double v = (floats[i] * 32767.0);
      // clamp and convert
      final int clamped = v.clamp(-32768.0, 32767.0).toInt();
      int16[i] = clamped;
    }

    final Uint8List bytes = int16.buffer.asUint8List();
    _receivedChunks.add(bytes);
    _player.uint8ListSink?.add(bytes);
    Logger().v(
      "✅ Stored audio chunk: ${bytes.length} bytes (total chunks: ${_receivedChunks.length})",
    );
  }

  // OTHER FUNCTIONS
  @override
  void onClose() {
    disposePlayer();
    super.onClose();
  }

  Future<void> disposePlayer() async {
    Logger().d("disposePlayer called");
    // Close ws subscription & sink
    await _channelSub?.cancel();
    await _channel?.sink.close();
    _channel = null;

    // Close flutter_sound player
    await _player.stopPlayer();
    await _player.closePlayer();

    // reset state
    isLoading.value = false;
    isSpeaking.value = false;
    _receivedChunks.clear();
  }

  Future<void> stopPlayer() async {
    Logger().d("stopPlayer called");
    await _player.stopPlayer();
    isSpeaking.value = false;
  }

  Future<void> startPlayer() async {
    Logger().d("startPlayer called");
  }

  void clear() {
    Logger().d("clear called");
    _receivedChunks.clear();
  }

  Future<void> speak(
    String text,
    String langCode,
    String langName,
    String langGender,
  ) async {
    // filePath.value = "dummy";
    if (_channel == null) {
      Logger().w("❌ Cannot speak: WS channel not connected");
      return;
    }

    final message = {"type": "text_message", "text": text, "voice_id": 0};

    Logger().i("➡️ Sending text_message: $message");
    _channel!.sink.add(jsonEncode(message));

    // Mark loading until we hear back from server
    isLoading.value = true;
    isSpeaking.value = false;
  }

  void setOnCompletionHandler(Function handler) {
    onCompletionHandler = handler;
  }
}
