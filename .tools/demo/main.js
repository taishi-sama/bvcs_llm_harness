export function main(args) {
  const listing = execute("fs_ls", {});
  return "demo:" + (args.message ?? "") + "|" + listing;
}
