package ai.liquid.browsercontrol

/**
 * 运行时 URL 配置。
 *
 * 通过切换 BASE_URL 注释，在本地 Docker（模拟器/物理设备）和远程之间切换：
 *
 * - 模拟器    : http://10.0.2.2:8080   (10.0.2.2 = Android 模拟器内置宿主机地址)
 * - 物理设备  : http://192.168.x.x:8080 (替换为宿主机 LAN IP，用 hostname -I 查询)
 * - 远程      : https://miniwob.farama.org/environments
 *
 * 启动 Docker 服务:
 *   cd examples/browser-control/docker/miniwob
 *   docker compose up -d --build
 */
object AppConfig {

    // ---- 取消注释其中一行 ----
    private const val BASE_URL = "http://192.168.3.64:8080"                        // 模拟器
    // private const val BASE_URL = "http://192.168.1.100:8080"                // 物理设备（改为实际IP）
    // private const val BASE_URL = "https://miniwob.farama.org/environments"  // 远程

    /**
     * 返回 MiniWoB++ 任务的完整 URL。
     *
     * - 本地 Docker: http://[host]:8080/miniwob/{task}.html
     * - 远程        : https://miniwob.farama.org/environments/{task}/
     */
    fun taskUrl(task: String = "click-button"): String =
        if (BASE_URL.contains("farama.org")) "$BASE_URL/$task/"
        else "$BASE_URL/miniwob/$task.html"
}
