# Android项目验证指南

> **项目**: BrowserControlDemo  
> **验证日期**: 2026-02-22  
> **验证状态**: ✅ 通过

---

## 快速验证

### 方法1: 使用验证脚本（推荐）

```bash
cd /home/tony/project/cookbook/examples/browser-control/android/BrowserControlDemo
./verify_project.sh
```

**验证项目**:
- ✅ 项目结构完整性（14个目录）
- ✅ 配置文件存在性（18个文件）
- ✅ 关键配置正确性（包名、SDK、依赖）
- ✅ Gradle Wrapper就绪
- ✅ Java环境可用

---

## 详细验证步骤

### 1. 文件结构验证

```bash
# 检查所有配置文件
find . -type f \( -name "*.kts" -o -name "*.xml" -o -name "*.kt" \) | sort

# 预期输出（14个文件）:
# ./app/build.gradle.kts
# ./app/proguard-rules.pro
# ./app/src/main/AndroidManifest.xml
# ./app/src/main/java/ai/liquid/browsercontrol/MainActivity.kt
# ./app/src/main/res/raw/system_prompt.txt
# ./app/src/main/res/values/strings.xml
# ./app/src/main/res/values/themes.xml
# ./app/src/main/res/xml/backup_rules.xml
# ./app/src/main/res/xml/data_extraction_rules.xml
# ./build.gradle.kts
# ./gradle.properties
# ./gradle/wrapper/gradle-wrapper.properties
# ./README.md
# ./settings.gradle.kts
```

### 2. Gradle配置验证

```bash
# 检查Gradle版本
./gradlew --version

# 预期输出:
# Gradle 8.5
# Kotlin: 1.9.10
# JVM: 17.0.18 或 21.0.8 (OpenJDK)
```

### 3. 查看可用任务

```bash
# 列出所有Gradle任务（首次运行会下载依赖）
./gradlew tasks --all

# 主要任务:
#   assembleDebug - 构建Debug APK
#   assembleRelease - 构建Release APK
#   installDebug - 安装Debug版本
#   clean - 清理构建产物
```

### 4. 依赖验证

```bash
# 查看依赖树（需要GitHub凭证）
./gradlew :app:dependencies

# 注意: LeapSDK需要GitHub Personal Access Token
```

### 5. 语法检查

```bash
# 检查Kotlin语法
./gradlew :app:compileDebugKotlin

# 注意: 首次运行会下载大量依赖（约500MB+）
```

---

## Gradle构建测试

### ⚠️ 前置条件

**配置GitHub凭证**（用于访问LeapSDK私有仓库）:

**方法1: 环境变量**
```bash
export GITHUB_USERNAME="your_username"
export GITHUB_TOKEN="your_github_pat"
```

**方法2: gradle.properties**
```bash
echo "gpr.user=your_username" >> ~/.gradle/gradle.properties
echo "gpr.token=your_github_pat" >> ~/.gradle/gradle.properties
```

**获取Token**:
1. GitHub → Settings → Developer settings
2. Personal access tokens → Tokens (classic)
3. Generate new token
4. 勾选 `read:packages` 权限

### 构建命令

```bash
# 1. 清理构建
./gradlew clean

# 2. 构建Debug APK
./gradlew assembleDebug

# 成功输出:
# BUILD SUCCESSFUL in 2m 15s
# APK: app/build/outputs/apk/debug/app-debug.apk
```

---

## 验证结果

### ✅ 基础验证（无需网络）

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 目录结构 | ✅ | 14个目录完整 |
| 配置文件 | ✅ | 18个文件存在 |
| Gradle Wrapper | ✅ | gradlew可执行 |
| wrapper.jar | ✅ | 63KB已下载 |
| 包名配置 | ✅ | ai.liquid.browsercontrol |
| SDK配置 | ✅ | min=31, target=34 |
| 架构配置 | ✅ | arm64-v8a only |
| Java版本 | ✅ | OpenJDK 17.0.18 |

### ⏳ 高级验证（需要网络+凭证）

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 依赖下载 | ⏳ | 需要GitHub Token |
| Kotlin编译 | ⏳ | 需要首次同步 |
| APK构建 | ⏳ | 需要完整依赖 |

---

## 常见问题

### Q1: gradlew: command not found

**原因**: gradlew没有执行权限

**解决**:
```bash
chmod +x gradlew
```

### Q2: Could not find ai.liquid.leap:leap-sdk

**原因**: 未配置GitHub凭证

**解决**: 参考上方"配置GitHub凭证"部分

### Q3: Unsupported class file major version 61

**原因**: Java版本不匹配

**解决**: 安装JDK 17
```bash
sudo apt install openjdk-17-jdk
```

### Q4: Gradle下载缓慢

**原因**: 网络问题

**解决**: 配置Gradle镜像
```bash
# 编辑 init.gradle
mkdir -p ~/.gradle/
cat > ~/.gradle/init.gradle << 'EOF'
allprojects {
    repositories {
        maven { url 'https://maven.aliyun.com/repository/public/' }
        maven { url 'https://maven.aliyun.com/repository/google/' }
    }
}
EOF
```

---

## 下一步操作

### 1. 在Android Studio中打开（推荐）

```bash
# 启动Android Studio并打开项目
android-studio /home/tony/project/cookbook/examples/browser-control/android/BrowserControlDemo
```

**首次打开会自动**:
- 下载Gradle依赖（约500MB）
- 索引项目文件
- 配置Android SDK

### 2. 命令行构建

```bash
# 配置凭证后执行
./gradlew assembleDebug

# 预期耗时: 2-5分钟（首次）
# 输出: app/build/outputs/apk/debug/app-debug.apk
```

### 3. 安装到设备

```bash
# 确保设备已连接并启用USB调试
adb devices

# 安装APK
./gradlew installDebug

# 或手动安装
adb install app/build/outputs/apk/debug/app-debug.apk
```

---

## 项目状态总结

### ✅ 已完成
- 项目结构创建
- Gradle配置完成
- 依赖声明完成
- 基础MainActivity实现
- 资源文件配置
- 文档编写

### 🎯 当前阶段
**阶段二: Android项目初始化** - ✅ 完成

### 📋 后续阶段
- **阶段三**: 核心组件实现（ModelRunner, ActionParser, BrowserView）
- **阶段四**: UI完整实现
- **阶段五**: MiniWoB任务集成测试

---

## 验证工具文件

- **验证脚本**: [verify_project.sh](verify_project.sh) - 快速检查工具
- **项目文档**: [README.md](README.md) - 开发指南
- **部署计划**: [../../../docs/browser-control-android-deploy.md](../../../docs/browser-control-android-deploy.md)

---

**最后更新**: 2026-02-22  
**验证者**: AI Agent  
**结论**: ✅ 项目配置正确，可以进入下一阶段开发
