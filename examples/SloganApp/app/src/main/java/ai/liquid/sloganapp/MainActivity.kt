package ai.liquid.sloganapp

import android.os.Bundle
import android.view.ViewGroup
import android.view.View
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updateLayoutParams
import android.widget.Button
import android.widget.EditText
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.viewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private val viewModel: SloganViewModel by viewModels()
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.main_activity_layout)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main_activity_layout)) { view, window ->
            val insets = window.getInsets(WindowInsetsCompat.Type.systemBars())
            view.updateLayoutParams<ViewGroup.MarginLayoutParams> {
                leftMargin = insets.left; rightMargin = insets.right
                topMargin = insets.top; bottomMargin = insets.bottom
            }
            window
        }
        val input = findViewById<EditText>(R.id.user_input)
        val button = findViewById<Button>(R.id.generate_button)
        val output = findViewById<TextView>(R.id.generated_text)
        val status = findViewById<TextView>(R.id.model_status)
        status.visibility = View.VISIBLE
        val scroll = findViewById<ScrollView>(R.id.generated_text_scrollview)
        button.setOnClickListener {
            if (viewModel.generation.value.busy) viewModel.stop()
            else viewModel.generate("Suggest slogans for this business: ${input.text.toString().trim()}".takeIf { input.text.isNotBlank() }.orEmpty())
        }
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.generation.collect { state ->
                    status.text = state.error ?: state.status
                    output.text = state.output
                    button.text = if (state.busy) "Stop" else "Generate"
                    input.isEnabled = !state.busy
                    scroll.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) }
                }
            }
        }
    }
}
