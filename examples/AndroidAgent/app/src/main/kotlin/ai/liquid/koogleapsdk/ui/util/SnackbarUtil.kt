package ai.liquid.koogleapsdk.ui.util

import androidx.compose.material3.SnackbarHostState

object SnackbarUtil {
    lateinit var snackbarHostState: SnackbarHostState

    suspend fun showSnackbar(
        message: String,
        actionLabel: String? = null
    ) {
        snackbarHostState.showSnackbar(
            message = message,
            actionLabel = actionLabel
        )
    }
}