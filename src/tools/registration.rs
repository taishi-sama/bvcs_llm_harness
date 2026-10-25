//! Uniform handler registration for inline tool modules.
//!
//! Every module that implements synchronous tools ends with an invocation of
//! [`tool_handlers!`]. It generates the `handler(name)` lookup consumed by
//! `catalog::execution`, so the dispatcher is one uniform chain over modules
//! instead of special cases per tool.

/// Generate the `handler(name) -> Option<ToolFn>` lookup for one module.
///
/// All registered functions share the [`ToolFn`](crate::tools::registry::ToolFn)
/// signature, `fn(&mut ToolWorld, &Value, &CallContext) -> (String, bool)`, so
/// the macro only maps names to functions.
///
/// ```ignore
/// tool_handlers! {
///     "fs_read" => fs_read,
///     "fs_write" => fs_write,
/// }
/// ```
macro_rules! tool_handlers {
    ($($name:literal => $handler:ident),* $(,)?) => {
        /// Handler lookup used by `catalog::execution`.
        pub(crate) fn handler(name: &str) -> Option<$crate::tools::registry::ToolFn> {
            Some(match name {
                $($name => $handler,)*
                _ => return None,
            })
        }
    };
}

pub(crate) use tool_handlers;
