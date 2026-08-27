import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:omi/backend/preferences.dart';
import 'package:omi/l10n/app_localizations.dart';
import 'package:omi/models/custom_stt_config.dart';
import 'package:omi/models/stt_provider.dart';
import 'package:omi/pages/settings/transcription_settings_page.dart';
import 'package:omi/providers/capture_provider.dart';

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('Gemini 3.5 Transcribe Live BYOK UI renders on Android device with new defaults', (tester) async {
    const apiKey = String.fromEnvironment('GEMINI_API_KEY', defaultValue: 'test-api-key');

    SharedPreferences.setMockInitialValues({});
    await SharedPreferencesUtil.init();

    // 1. Configure Gemini Live provider
    const geminiLiveConfig = CustomSttConfig(
      provider: SttProvider.geminiLive,
      apiKey: apiKey,
      model: 'gemini-3.5-transcribe-live-preview',
      language: 'en',
    );
    await SharedPreferencesUtil().saveCustomSttConfig(geminiLiveConfig);
    await SharedPreferencesUtil().saveConfigForProvider(SttProvider.geminiLive, geminiLiveConfig);

    final captureProvider = CaptureProvider();
    addTearDown(captureProvider.dispose);
    tester.view.physicalSize = const Size(1080, 1920);
    tester.view.devicePixelRatio = 2.625;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    // 2. Pump the Transcription Settings UI on device
    await tester.pumpWidget(
      ChangeNotifierProvider<CaptureProvider>.value(
        value: captureProvider,
        child: const MaterialApp(
          localizationsDelegates: [
            AppLocalizations.delegate,
            GlobalMaterialLocalizations.delegate,
            GlobalWidgetsLocalizations.delegate,
            GlobalCupertinoLocalizations.delegate,
          ],
          supportedLocales: AppLocalizations.supportedLocales,
          home: TranscriptionSettingsPage(),
        ),
      ),
    );
    await tester.pumpAndSettle(const Duration(milliseconds: 500));

    debugPrint('[ON-DEVICE TEST] TranscriptionSettingsPage rendered successfully on Android');

    // 3. Verify on-device UI elements
    expect(find.byType(TranscriptionSettingsPage), findsOneWidget);
    expect(find.text('Google Gemini Live'), findsWidgets);

    // Verify model defaults and request config on device
    final resolvedConfig = CustomSttConfig.fromJson({'provider': 'geminiLive', 'api_key': apiKey});
    expect(resolvedConfig.effectiveModel, 'gemini-3.5-transcribe-live-preview');
    expect(resolvedConfig.requestConfig['params']['model'], 'gemini-3.5-transcribe-live-preview');

    debugPrint('[ON-DEVICE TEST] On-device resolved effectiveModel: ${resolvedConfig.effectiveModel}');
    debugPrint('[ON-DEVICE TEST] On-device resolved socket params: ${resolvedConfig.requestConfig['params']}');

    // 4. Convert surface and take on-device screenshot
    await binding.convertFlutterSurfaceToImage();
    await tester.pumpAndSettle();
    await binding.takeScreenshot('gemini_35_transcription_settings_android');
    debugPrint('[ON-DEVICE TEST] Screenshot captured: gemini_35_transcription_settings_android');

    debugPrint('[ON-DEVICE TEST] ALL ON-DEVICE ASSERTIONS PASSED SUCCESSFULLY!');
  });
}
