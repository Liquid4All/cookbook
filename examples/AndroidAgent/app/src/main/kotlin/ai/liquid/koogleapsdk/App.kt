package ai.liquid.koogleapsdk

import android.app.Application
import android.content.Context

class App : Application() {

    override fun onCreate() {
        super.onCreate()
        instance = this@App
    }

    companion object {
        internal lateinit var instance: App
            private set
        val context: Context by lazy { instance.applicationContext }
    }
}