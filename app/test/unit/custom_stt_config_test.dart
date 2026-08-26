import 'package:flutter_test/flutter_test.dart';
import 'package:omi/models/custom_stt_config.dart';
import 'package:omi/models/stt_provider.dart';

void main() {
  group('Gemini 3.5 Transcribe defaults', () {
    // Guards the BYOK model upgrade: batch/live Gemini providers must default to
    // the Gemini 3.5 Transcribe models. Expected IDs come from the Gemini 3.5
    // Transcribe release on the Gemini Enterprise Agent Platform.
    // https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-transcribe

    test('batch provider defaults to gemini-3.5-transcribe-preview', () {
      final config = CustomSttConfig.fromJson({'provider': 'gemini'});
      expect(config.effectiveModel, 'gemini-3.5-transcribe-preview');
    });

    test('live provider defaults to gemini-3.5-transcribe-live-preview', () {
      final config = CustomSttConfig.fromJson({'provider': 'geminiLive'});
      expect(config.effectiveModel, 'gemini-3.5-transcribe-live-preview');
    });

    test('batch supported models advertise the 3.5 preview first', () {
      final supported = SttProviderConfig.get(SttProvider.gemini).supportedModels;
      expect(supported.first, 'gemini-3.5-transcribe-preview');
      expect(supported, contains('gemini-3.5-transcribe-preview'));
    });

    test('live supported models advertise the 3.5 live preview first', () {
      final supported = SttProviderConfig.get(SttProvider.geminiLive).supportedModels;
      expect(supported.first, 'gemini-3.5-transcribe-live-preview');
      expect(supported, contains('gemini-3.5-transcribe-live-preview'));
    });

    test('batch request URL embeds the 3.5 preview model when none is chosen', () {
      final config = CustomSttConfig.fromJson({'provider': 'gemini', 'api_key': 'test-key'});
      final url = config.requestConfig['url'] as String;
      expect(url, contains('models/gemini-3.5-transcribe-preview:generateContent'));
    });

    test('live request params carry the 3.5 live preview model when none is chosen', () {
      final config = CustomSttConfig.fromJson({'provider': 'geminiLive', 'api_key': 'test-key'});
      final params = config.requestConfig['params'] as Map<String, dynamic>;
      expect(params['model'], 'gemini-3.5-transcribe-live-preview');
    });

    test('a user-selected model overrides the 3.5 default', () {
      final config = CustomSttConfig.fromJson({'provider': 'gemini', 'model': 'gemini-2.5-flash'});
      expect(config.effectiveModel, 'gemini-2.5-flash');
      expect(config.requestConfig['url'] as String, contains('models/gemini-2.5-flash:generateContent'));
    });
  });

  group('CustomSttConfig raw audio forwarding', () {
    test('legacy configs keep forwarding raw audio to Omi', () {
      final config = CustomSttConfig.fromJson({'provider': 'customLive'});

      expect(config.toJson()['send_raw_audio_to_omi'], isTrue);
    });

    test('disabled forwarding survives a JSON round trip', () {
      final config = CustomSttConfig.fromJson({'provider': 'customLive', 'send_raw_audio_to_omi': false});

      expect(config.toJson()['send_raw_audio_to_omi'], isFalse);
    });

    test('forwarding policy participates in the config identity', () {
      final forwarding = CustomSttConfig.fromJson({'provider': 'customLive', 'send_raw_audio_to_omi': true});
      final transcriptOnly = CustomSttConfig.fromJson({'provider': 'customLive', 'send_raw_audio_to_omi': false});

      expect(forwarding.sttConfigId, isNot(transcriptOnly.sttConfigId));
    });
  });
}
