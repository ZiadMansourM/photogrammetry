//
//  LogoView.swift
//  Tridify
//
//  Created by Maged Alosali on 28/04/2023.
//

import SwiftUI

struct LogoView: View {
    @Environment(\.colorScheme) private var colorScheme
    var body: some View {
        if colorScheme == .light {
            Image("icon-light")
                .resizable()
                .scaledToFit()
        }
        else {
            Image("icon-dark")
                .resizable()
                .scaledToFit()
        }
    }
}

struct LogoView_Previews: PreviewProvider {
    static var previews: some View {
        LogoView()
        LogoView()
            .preferredColorScheme(.dark)
    }
}
