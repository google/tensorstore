_ICONS_ATTRS = {
    "deps": attr.label_list(),
    "aliases": attr.string_dict(),
}

def _icons_impl(ctx):
    runfiles = []
    source_files = []
    for dep in ctx.attr.deps:
        if DefaultInfo not in dep:
            continue
        dep_runfiles = dep[DefaultInfo].default_runfiles
        if not dep_runfiles:
            continue
        source_files.append(dep_runfiles.files)
    for f in depset(transitive = source_files).to_list():
        for alias_source, alias_target in ctx.attr.aliases.items():
            idx = f.short_path.find("/" + alias_source)
            if idx == -1:
                continue
            symlink_name = alias_target + f.short_path[idx + 1 + len(alias_source):]
            symlink = ctx.actions.declare_file(symlink_name)
            runfiles.append(symlink)
            ctx.actions.symlink(output = symlink, target_file = f)

    return struct(
        providers = [
            DefaultInfo(
                runfiles = ctx.runfiles(
                    files = runfiles,
                ),
            ),
        ],
    )

icon_symlinks = rule(
    implementation = _icons_impl,
    attrs = _ICONS_ATTRS,
)
