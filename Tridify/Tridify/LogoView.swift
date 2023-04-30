//
//  LogoView.swift
//  Tridify
//
//  Created by Maged Alosali on 28/04/2023.
//

import SwiftUI

struct LogoView: View {
    
    @Environment(\.colorScheme) private var colorScheme
    
    private var isLightMode: Bool {
        colorScheme == .light
    }
    
    var body: some View {
        Image(isLightMode ? "icon-light" : "icon-dark")
            .resizable()
            .scaledToFit()
    }
}

struct LogoView_Previews: PreviewProvider {
    static var previews: some View {
        LogoView()
        LogoView()
            .preferredColorScheme(.dark)
    }
}
