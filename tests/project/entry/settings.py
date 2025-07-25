UNFAZED_SETTINGS = {
    "ROOT_URLCONF": "entry.routes",
    "INSTALLED_APPS": ["event"],
    "MIDDLEWARE": [
        "unfazed.middleware.internal.common.CommonMiddleware",
    ],
    "LIFESPAN": ["event.mcp.EventMCPLifespan"],
}
