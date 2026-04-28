#!/bin/bash

echo "=========================================="
echo "  Android项目验证脚本"
echo "  项目: BrowserControlDemo"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 计数器
PASSED=0
FAILED=0

# 验证函数
check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}❌ $1${NC}"
        ((FAILED++))
        return 1
    fi
}

echo "1️⃣  检查项目结构..."
echo "----------------------------"

# 检查关键目录
[ -d "app/src/main/java/ai/liquid/browsercontrol" ]
check "Kotlin源码目录存在"

[ -d "app/src/main/res" ]
check "资源目录存在"

[ -d "gradle/wrapper" ]
check "Gradle Wrapper目录存在"

echo ""
echo "2️⃣  检查关键配置文件..."
echo "----------------------------"

[ -f "build.gradle.kts" ]
check "项目级build.gradle.kts存在"

[ -f "app/build.gradle.kts" ]
check "应用级build.gradle.kts存在"

[ -f "settings.gradle.kts" ]
check "settings.gradle.kts存在"

[ -f "app/src/main/AndroidManifest.xml" ]
check "AndroidManifest.xml存在"

[ -f "gradle.properties" ]
check "gradle.properties存在"

echo ""
echo "3️⃣  检查源文件..."
echo "----------------------------"

[ -f "app/src/main/java/ai/liquid/browsercontrol/MainActivity.kt" ]
check "MainActivity.kt存在"

[ -f "app/src/main/res/raw/system_prompt.txt" ]
check "system_prompt.txt存在"

[ -f "app/src/main/res/values/strings.xml" ]
check "strings.xml存在"

echo ""
echo "4️⃣  检查Gradle Wrapper..."
echo "----------------------------"

[ -f "gradlew" ] && [ -x "gradlew" ]
check "gradlew可执行文件存在"

[ -f "gradle/wrapper/gradle-wrapper.jar" ]
check "gradle-wrapper.jar存在"

[ -f "gradle/wrapper/gradle-wrapper.properties" ]
check "gradle-wrapper.properties存在"

echo ""
echo "5️⃣  检查配置内容..."
echo "----------------------------"

# 检查包名
grep -q "ai.liquid.browsercontrol" app/build.gradle.kts
check "包名配置正确 (ai.liquid.browsercontrol)"

# 检查最低SDK
grep -q "minSdk = 31" app/build.gradle.kts
check "最低SDK配置正确 (31)"

# 检查LeapSDK依赖
grep -q "ai.liquid.leap:leap-sdk" app/build.gradle.kts
check "LeapSDK依赖已配置"

# 检查Compose依赖
grep -q "androidx.compose" app/build.gradle.kts
check "Jetpack Compose依赖已配置"

# 检查arm64架构
grep -q "arm64-v8a" app/build.gradle.kts
check "ARM64架构配置正确"

echo ""
echo "6️⃣  文件权限检查..."
echo "----------------------------"

ls -l gradlew | grep -q "rwx"
check "gradlew有执行权限"

echo ""
echo "7️⃣  Java环境检查..."
echo "----------------------------"

if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | head -n 1)
    echo "   Java版本: $JAVA_VERSION"
    check "Java已安装"
else
    echo -e "${RED}   Java未安装${NC}"
    ((FAILED++))
fi

echo ""
echo "=========================================="
echo "  验证结果汇总"
echo "=========================================="
echo -e "${GREEN}✅ 通过: $PASSED${NC}"
echo -e "${RED}❌ 失败: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 所有检查通过！项目配置正确。${NC}"
    echo ""
    echo "📝 建议的下一步:"
    echo "   1. 配置GitHub凭证用于LeapSDK访问"
    echo "   2. 运行: ./gradlew tasks 查看可用任务"
    echo "   3. 运行: ./gradlew assembleDebug 构建应用"
    exit 0
else
    echo -e "${YELLOW}⚠️  有 $FAILED 项检查未通过，请修复后重试。${NC}"
    exit 1
fi
