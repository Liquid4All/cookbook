#import "AudioTapInstallerBridge.h"

static NSString * const LFMAudioTapInstallerErrorDomain = @"ai.liquid.telcotriage.audioTap";

BOOL LFMInstallAudioTapSafely(
    AVAudioNode *node,
    AVAudioNodeBus bus,
    AVAudioFrameCount bufferSize,
    AVAudioFormat *_Nullable format,
    LFMAudioTapBlock block,
    NSError *_Nullable *_Nullable error
) {
    @try {
        [node installTapOnBus:bus bufferSize:bufferSize format:format block:block];
        return YES;
    } @catch (NSException *exception) {
        if (error != nil) {
            NSString *reason = exception.reason ?: @"AVAudioEngine rejected the microphone input route.";
            NSString *message = [NSString stringWithFormat:@"Microphone input route failed: %@", reason];
            *error = [NSError errorWithDomain:LFMAudioTapInstallerErrorDomain
                                         code:1
                                     userInfo:@{
                                         NSLocalizedDescriptionKey: message,
                                         @"exceptionName": exception.name ?: @"NSException",
                                         @"exceptionReason": reason
                                     }];
        }
        return NO;
    }
}
