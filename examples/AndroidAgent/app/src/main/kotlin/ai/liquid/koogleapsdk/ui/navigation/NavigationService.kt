package ai.liquid.koogleapsdk.ui.navigation

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

interface NavigationService {
    val destinationFlow: StateFlow<Destination>

    fun navigateTo(destination: Destination)

    fun navigateBack()

    companion object {
        val Instance: NavigationService by lazy {
            NavigationServiceImpl()
        }
    }
}

private class NavigationServiceImpl() : NavigationService {
    private val backStack = ArrayDeque<Destination>().apply {
        add(Destination.ToolsList)
    }
    private val _destinationFlow = MutableStateFlow(backStack.last())
    override val destinationFlow: StateFlow<Destination>
        get() = _destinationFlow.asStateFlow()


    override fun navigateTo(destination: Destination) {
        backStack.add(destination)
        _destinationFlow.update {
            destination
        }
    }

    override fun navigateBack() {
        if (backStack.size > 1) {
            backStack.removeLastOrNull()
            _destinationFlow.update {
                backStack.last()
            }
        }
    }
}