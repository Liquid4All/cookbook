# BrowserControlDemo - Android Application

> **创建时间**: 2026-02-22  
> **当前阶段**: 阶段二 - Android项目初始化 ✅  
> **目标**: LFM2-350M browser-control模型Android端验证

## 项目信息

- **包名**: `ai.liquid.browsercontrol`
- **最低SDK**: 31 (Android 12)
- **目标SDK**: 34
- **架构**: arm64-v8a only
- **UI框架**: Jetpack Compose + Material3
- **推理引擎**: LeapSDK 0.9.7

## 目录结构

```
BrowserControlDemo/
├── app/
│   ├── build.gradle.kts          # 应用级Gradle配置
│   ├── proguard-rules.pro        # ProGuard规则
│   └── src/main/
│       ├── AndroidManifest.xml   # 应用清单
│       ├── java/ai/liquid/browsercontrol/
│       │   └── MainActivity.kt   # 主Activity
│       └── res/
│           ├── raw/
│           │   └── system_prompt.txt  # 系统Prompt
│           ├── values/
│           │   ├── strings.xml   # 字符串资源
│           │   └── themes.xml    # 主题配置
│           └── xml/
│               ├── backup_rules.xml
│               └── data_extraction_rules.xml
├── build.gradle.kts              # 项目级Gradle配置
├── settings.gradle.kts           # 项目设置
├── gradle.properties             # Gradle属性
└── gradle/wrapper/
    └── gradle-wrapper.properties # Gradle Wrapper配置

## 开发环境

### 必需
- **Android Studio**: Hedgehog (2023.1.1) 或更新
- **JDK**: 17 或 21
- **Gradle**: 8.5
- **Kotlin**: 1.9.10
- **Android Gradle Plugin**: 8.2.2

### LeapSDK访问

需要配置GitHub Package Registry凭证:

**方法1: 环境变量**
```bash
export GITHUB_USERNAME="your_username"
export GITHUB_TOKEN="your_personal_access_token"
```

**方法2: gradle.properties**
```properties
gpr.user=your_username
gpr.token=your_personal_access_token
```

## 构建步骤

### 1. 打开项目
```bash
cd /home/tony/project/cookbook/examples/browser-control/android/BrowserControlDemo
# 使用Android Studio打开此目录
```

### 2. 同步Gradle
```bash
./gradlew sync
```

### 3. 构建APK
```bash
./gradlew assembleDebug
# 输出: app/build/outputs/apk/debug/app-debug.apk
```

### 4. 安装到设备
```bash
./gradlew installDebug
# 或
adb install app/build/outputs/apk/debug/app-debug.apk
```

## 开发路线图

### ✅ 阶段二: Android项目初始化 (已完成)
- [x] 创建项目目录结构
- [x] 配置Gradle构建脚本
- [x] 配置AndroidManifest权限
- [x] 创建系统Prompt文件
- [x] 创建基础MainActivity

### ⏳ 阶段三: 核心组件实现 (待开始)
- [ ] 数据模型 (BrowserObservation, BrowserAction)
- [ ] ModelRunner (LeapSDK推理)
- [ ] ActionParser (解析模型输出)
- [ ] BrowserView (WebView封装)

### ⏳ 阶段四: UI实现 (待开始)
- [ ] Jetpack Compose完整界面
- [ ] 任务控制面板
- [ ] 浏览器视图
- [ ] 动作日志显示

### ⏳ 阶段五: 集成测试 (待开始)
- [ ] MiniWoB click-test任务
- [ ] 端到端验证

### ⏳ 阶段六: 文档与优化 (待开始)
- [ ] 技术文档
- [ ] 性能分析
- [ ] 优化建议

## 依赖项

### LeapSDK
```kotlin
implementation("ai.liquid.leap:leap-sdk:0.9.7")
implementation("ai.liquid.leap:leap-model-downloader:0.9.7")
```

### Jetpack Compose
```kotlin
implementation(platform("androidx.compose:compose-bom:2023.10.01"))
implementation("androidx.compose.ui:ui")
implementation("androidx.compose.material3:material3")
implementation("androidx.activity:activity-compose:1.8.2")
```

### 其他
- Kotlin Coroutines: 1.7.3
- AndroidX WebKit: 1.9.0

## 注意事项

1. **架构限制**: 仅支持arm64-v8a，不支持x86/x86_64
2. **模型文件**: GGUF模型需要手动放置到assets或下载
3. **网络权限**: 需要INTERNET权限用于WebView和模型推理
4. **存储权限**: Android 12+不再需要WRITE_EXTERNAL_STORAGE

## 参考资料

- **LeapSDK文档**: https://github.com/Liquid4All/LeapSDK-Examples
- **部署计划**: [../browser-control-android-deploy.md](../../../docs/browser-control-android-deploy.md)
- **训练文档**: [../browser-control-model-deploy.md](../../../docs/browser-control-model-deploy.md)

---

**创建者**: AI Agent  
**项目**: cookbook/examples/browser-control  
**更新日期**: 2026-02-22
