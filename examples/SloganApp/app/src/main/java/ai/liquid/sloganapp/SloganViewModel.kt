package ai.liquid.sloganapp

import android.app.Application
import ai.liquid.android.runtime.Models
import ai.liquid.android.runtime.TextTaskViewModel

class SloganViewModel(application: Application) : TextTaskViewModel(
    application, Models.text,
    "You are a marketing expert. Suggest engaging and creative slogans based on a business description provided by the user.",
)
