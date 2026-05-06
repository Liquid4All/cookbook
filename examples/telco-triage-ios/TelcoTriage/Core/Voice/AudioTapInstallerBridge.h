#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef void (^LFMAudioTapBlock)(AVAudioPCMBuffer *buffer, AVAudioTime *when);

BOOL LFMInstallAudioTapSafely(
    AVAudioNode *node,
    AVAudioNodeBus bus,
    AVAudioFrameCount bufferSize,
    AVAudioFormat *_Nullable format,
    LFMAudioTapBlock block,
    NSError *_Nullable *_Nullable error
);

NS_ASSUME_NONNULL_END
