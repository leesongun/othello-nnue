const game = @import("othello");

export fn flip(a: u64, b: u64, c: u8) u64 {
    const x = game{ .board = .{ a, b } };
    return x.flip(@intCast(u6, c));
}

export fn moves(a: u64, b: u64) u64 {
    const x = game{ .board = .{ a, b } };
    return x.moves();
}

export fn stable(a: u64, b: u64) u64 {
    const x = game{ .board = .{ a, b } };
    return x.stable();
}
