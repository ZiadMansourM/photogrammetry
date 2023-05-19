//
//  Color-Theme.swift
//  Tridify
//
//  Created by Maged Alosali on 29/04/2023.
//

import SwiftUI

extension ShapeStyle where Self == Color {
    static var darkHeadline: Color {
        .white.opacity(0.65)
    }
    static var lightHeadline: Color {
        .black.opacity(0.65)
    }
    
    static var darkButton: Color {
        .white.opacity(0.98)
    }
    
    static var lightButton: Color {
        black.opacity(0.9)
    }
    
    static var link: Color {
        .blue
    }
    
    static var emptyImage: Color {
        Color(red: 0.45, green: 0.45, blue: 0.45).opacity(0.4)
    }
}
